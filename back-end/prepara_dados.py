import pandas as pd
import glob
import os
import sqlite3
import unidecode

def normalizar_nome_municipio(nome):
    """
    Normaliza o nome do município removendo acentos, convertendo para maiúsculas e
    removendo espaços extras para garantir a correspondência correta.
    """
    if isinstance(nome, str):
        return unidecode.unidecode(nome).upper().strip()
    return None

def processa_arquivos_inmet(pasta_inmet):
    """
    Lê e consolida todos os arquivos CSV do INMET em um único DataFrame,
    lidando com nomes de colunas inconsistentes e extraindo o código da estação
    do cabeçalho do arquivo.
    """
    print("Iniciando a leitura e consolidação dos arquivos do INMET...")

    lista_dfs_inmet = []

    arquivos_inmet = glob.glob(os.path.join(pasta_inmet, 'INMET_*.CSV')) + \
                     glob.glob(os.path.join(pasta_inmet, 'INMET_*.csv'))

    if not arquivos_inmet:
        print("AVISO: Nenhum arquivo INMET encontrado na pasta.")
        return pd.DataFrame()

    for arquivo in arquivos_inmet:
        try:
            print(f"Processando arquivo: {os.path.basename(arquivo)}")

            # 1. Extrai o código da estação do cabeçalho do arquivo
            codigo_estacao = None
            with open(arquivo, 'r', encoding='latin1') as f:
                for linha in f:
                    if 'CODIGO (WMO):;' in linha:
                        codigo_estacao = linha.split(';')[1].strip()
                        break

            if not codigo_estacao:
                print(f"AVISO: Código de estação não encontrado no cabeçalho do arquivo {os.path.basename(arquivo)}. Pulando.")
                continue

            # 2. Lê o arquivo CSV a partir da linha 9, que é onde os dados começam
            df = pd.read_csv(arquivo, sep=';', decimal=',', encoding='latin1', skiprows=8, na_values=['-9999'])
            df.columns = df.columns.str.strip().str.upper()

            # 3. Adiciona a coluna com o código da estação que foi extraído
            df['CODIGOESTACAO'] = codigo_estacao

            # 4. Define o mapeamento de colunas
            coluna_map = {
                'DATA (YYYY-MM-DD)': 'DATA',
                'HORA (UTC)': 'HORA',
                'UMIDADE RELATIVA DO AR, HORARIA (%)': 'UMIDADE',
                'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'TEMPERATURA',
                'VENTO, VELOCIDADE HORARIA (m/s)': 'VENTO',
                'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'PRECIPITACAO'
            }

            # 5. Renomeia apenas as colunas que realmente existem no DataFrame
            df.rename(columns={old_name: new_name for old_name, new_name in coluna_map.items() if old_name in df.columns}, inplace=True)

            # 6. Seleciona apenas as colunas que são de interesse e estão presentes
            colunas_interesse = [
                'CODIGOESTACAO', 'DATA', 'HORA', 'TEMPERATURA',
                'UMIDADE', 'VENTO', 'PRECIPITACAO'
            ]

            # Adiciona colunas ausentes com valores nulos para manter a consistência
            for col in colunas_interesse:
                if col not in df.columns:
                    df[col] = pd.NA

            df = df[colunas_interesse]

            # 7. Converte as colunas para o tipo correto e trata valores ausentes
            df['DATA'] = pd.to_datetime(df['DATA'])
            
            numeric_cols = ['TEMPERATURA', 'UMIDADE', 'VENTO', 'PRECIPITACAO']
            for col in numeric_cols:
                # Converte para float, valores inválidos se tornam NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Preenche valores NaN com a média da coluna para evitar erros de treinamento
                df[col] = df[col].fillna(df[col].mean())

            lista_dfs_inmet.append(df)

        except Exception as e:
            print(f"ERRO: Falha ao processar o arquivo {os.path.basename(arquivo)}: {e}")

    if lista_dfs_inmet:
        return pd.concat(lista_dfs_inmet, ignore_index=True)
    return pd.DataFrame()

def main():
    pasta_dados = os.path.join(os.path.dirname(__file__), 'dados')

    # --- 1. Carregar e processar a base de dados de inundações da ANA ---
    print("Processando base de dados de inundações da ANA...")
    caminho_inundacao = os.path.join(pasta_dados, 'ana_inundacao.csv')
    df_inundacao = pd.read_csv(caminho_inundacao)
    df_inundacao['CHEIAS_201'] = df_inundacao['CHEIAS_201'].apply(lambda x: 1 if x > 0 else 0)
    print("Processamento da base da ANA concluído.")

    # --- 2. Criar mapeamento de estação do INMET para código IBGE ---
    print("Criando mapeamento de estações INMET para códigos IBGE...")
    caminho_catalogo = os.path.join(pasta_dados, 'catalogoestacoesautomaticas.csv')
    df_catalogo = pd.read_csv(caminho_catalogo, delimiter=';')

    # Mapeamento de nomes normalizados
    df_catalogo['DC_NOME_NORMALIZADO'] = df_catalogo['DC_NOME'].apply(normalizar_nome_municipio)
    df_inundacao['NM_MUNICIP_NORMALIZADO'] = df_inundacao['NM_MUNICIP'].apply(normalizar_nome_municipio)

    # Junção inicial para mapear estação -> município -> código IBGE
    mapeamento_estacao_municipio = pd.merge(
        df_catalogo[['CD_ESTACAO', 'DC_NOME_NORMALIZADO']],
        df_inundacao[['CD_GEOCMU', 'NM_MUNICIP_NORMALIZADO']].drop_duplicates(),
        left_on='DC_NOME_NORMALIZADO',
        right_on='NM_MUNICIP_NORMALIZADO',
        how='inner'
    )
    mapeamento_estacao_municipio.drop(columns=['NM_MUNICIP_NORMALIZADO'], inplace=True)
    print("Mapeamento de código IBGE criado.")

    # --- 3. Unir os dados do INMET com o mapeamento e a inundação da ANA ---
    caminho_pasta_inmet = os.path.join(pasta_dados, 'inmet_data')
    df_inmet = processa_arquivos_inmet(pasta_inmet=caminho_pasta_inmet)
    if df_inmet.empty:
        print("Finalizando o script: nenhum dado do INMET processado.")
        return None

    df_inmet_com_ibge = pd.merge(
        df_inmet,
        mapeamento_estacao_municipio,
        left_on='CODIGOESTACAO',
        right_on='CD_ESTACAO',
        how='left'
    )

    df_final = pd.merge(
        df_inmet_com_ibge,
        df_inundacao[['CD_GEOCMU', 'CHEIAS_201']].drop_duplicates(),
        on='CD_GEOCMU',
        how='left'
    )

    df_final.rename(columns={'CHEIAS_201': 'Enchente'}, inplace=True)
    df_final['Enchente'] = df_final['Enchente'].fillna(0).astype(int)

    df_final = pd.merge(
        df_final,
        df_inundacao[['CD_GEOCMU', 'NM_MUNICIP']].drop_duplicates(),
        on='CD_GEOCMU',
        how='left'
    )
    df_final.rename(columns={'NM_MUNICIP': 'municipio'}, inplace=True)

    df_final.drop(columns=['DC_NOME_NORMALIZADO', 'CD_ESTACAO', 'CD_GEOCMU'], inplace=True)

    print("SUCESSO: Datasets combinados com sucesso!")

    # --- 4. Salvar o dataset final no SQLite ---
    print("Salvando o dataset final no banco de dados...")
    conn = sqlite3.connect('database.db')
    df_final.to_sql('clima_historico', conn, if_exists='replace', index=False)
    conn.close()
    print("Dados salvos em 'database.db'.")
    return "Dados preparados e salvos com sucesso."

if __name__ == "__main__":
    main()