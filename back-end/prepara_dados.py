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
    Lê e consolida todos os arquivos CSV do INMET em um único DataFrame.
    """
    print("Iniciando a leitura e consolidação dos arquivos do INMET...")
    
    lista_dfs_inmet = []
    
    # Busca por arquivos com a extensão em maiúsculas ou minúsculas
    arquivos_inmet = glob.glob(os.path.join(pasta_inmet, 'INMET_*.CSV')) + \
                     glob.glob(os.path.join(pasta_inmet, 'INMET_*.csv'))
    
    if not arquivos_inmet:
        print("AVISO: Nenhum arquivo INMET encontrado na pasta.")
        return pd.DataFrame()
        
    for arquivo in arquivos_inmet:
        try:
            df = pd.read_csv(arquivo, sep=';', decimal=',', encoding='latin1', skiprows=8)
            df.columns = df.columns.str.strip().str.upper()
            
            # Padronizando as colunas para facilitar a união
            if 'CODIGO ESTACAO' in df.columns:
                df = df.rename(columns={'CODIGO ESTACAO': 'CODIGOESTACAO'})
            
            # Adicionando a coluna de DataHora
            if 'DATA (YYYY-MM-DD)' in df.columns and 'HORA (UTC)' in df.columns:
                df['DATAHORA'] = pd.to_datetime(df['DATA (YYYY-MM-DD)'] + ' ' + df['HORA (UTC)'], format='%Y-%m-%d %H:%M')
            
            # Adiciona a coluna 'CodigoEstacao' se ainda não existir
            if 'CODIGOESTACAO' not in df.columns:
                nome_arquivo = os.path.basename(arquivo)
                codigo_estacao = nome_arquivo.split('_')[2].strip() if len(nome_arquivo.split('_')) > 2 else ''
                df['CODIGOESTACAO'] = codigo_estacao
            
            lista_dfs_inmet.append(df)
            print(f"SUCESSO: Arquivo {os.path.basename(arquivo)} processado.")

        except Exception as e:
            print(f"ERRO ao processar o arquivo {os.path.basename(arquivo)}: {e}")
            continue

    if not lista_dfs_inmet:
        return pd.DataFrame()
    
    df_inmet = pd.concat(lista_dfs_inmet, ignore_index=True)
    
    # Renomeando colunas para um padrão unificado
    df_inmet = df_inmet.rename(columns={
        'PRECIPITAÇÃO TOTAL, HORÁRIO (MM)': 'PRECIPITACAO',
        'UMIDADE RELATIVA DO AR, HORARIA (%)': 'UMIDADE',
        'VENTO, VELOCIDADE HORARIA (M/S)': 'VENTO',
        'TEMPERATURA DO AR - BULBO SECO, HORARIA (C)': 'TEMPERATURA'
    })
    
    # Selecionando apenas as colunas relevantes
    colunas_relevantes_inmet = ['CODIGOESTACAO', 'DATAHORA', 'PRECIPITACAO', 'UMIDADE', 'VENTO', 'TEMPERATURA']
    df_inmet = df_inmet[df_inmet.columns.intersection(colunas_relevantes_inmet)]
    
    return df_inmet

def processa_arquivos_ana(pasta_ana):
    """
    Lê os arquivos da ANA e retorna como DataFrames.
    """
    print("Lendo arquivos da ANA...")
    caminho_inundacao = os.path.join(pasta_ana, 'ana_inundacao.csv')
    caminho_vulnerabilidade = os.path.join(pasta_ana, 'ana_vulnerabilidade.csv')
    
    try:
        df_inundacao = pd.read_csv(caminho_inundacao)
        print("SUCESSO: Arquivo ana_inundacao.csv lido.")
    except FileNotFoundError:
        print(f"ERRO: Arquivo {caminho_inundacao} não encontrado.")
        df_inundacao = None
    
    try:
        df_vulnerabilidade = pd.read_csv(caminho_vulnerabilidade)
        print("SUCESSO: Arquivo ana_vulnerabilidade.csv lido.")
    except FileNotFoundError:
        print(f"ERRO: Arquivo {caminho_vulnerabilidade} não encontrado.")
        df_vulnerabilidade = None

    return df_inundacao, df_vulnerabilidade

def prepara_datasets_para_treinamento():
    """
    Função principal que orquestra o carregamento, união e salvamento dos datasets.
    """
    print("Iniciando o preparo dos datasets...")

    pasta_dados = os.path.join(os.path.dirname(__file__), 'dados')
    if not os.path.exists(pasta_dados):
        print(f"ERRO CRÍTICO: Pasta '{pasta_dados}' não encontrada.")
        return None

    # --- 1. Carregar todos os arquivos necessários ---
    # Caminho do arquivo de catálogo, nomeado de acordo com a sua correção anterior
    caminho_catalogo = os.path.join(pasta_dados, 'catalogoestacoesautomaticas.csv')
    
    try:
        df_catalogo = pd.read_csv(caminho_catalogo, delimiter=';', encoding='utf-8')
        df_inundacao, _ = processa_arquivos_ana(pasta_ana=pasta_dados)
    except FileNotFoundError as e:
        print(f"ERRO CRÍTICO: Arquivo não encontrado: {e}")
        return None

    if df_inundacao is None:
        return None

    # --- 2. Normalizar e criar o mapeamento ---
    df_catalogo['DC_NOME_NORMALIZADO'] = df_catalogo['DC_NOME'].apply(normalizar_nome_municipio)
    df_inundacao['NM_MUNICIP_NORMALIZADO'] = df_inundacao['NM_MUNICIP'].apply(normalizar_nome_municipio)

    mapeamento_estacao_municipio = pd.merge(
        df_catalogo[['CD_ESTACAO', 'DC_NOME_NORMALIZADO']],
        df_inundacao[['NM_MUNICIP_NORMALIZADO', 'CD_GEOCMU']].drop_duplicates(),
        left_on='DC_NOME_NORMALIZADO',
        right_on='NM_MUNICIP_NORMALIZADO',
        how='inner'
    )
    
    if mapeamento_estacao_municipio.empty:
        print("ERRO CRÍTICO: Não foi possível criar o mapeamento. Verifique se os nomes dos municípios correspondem.")
        return None

    print("SUCESSO: Arquivo de mapeamento de estação INMET para código IBGE criado.")

    # --- 3. Unir os dados do INMET com o mapeamento e a inundação da ANA ---
    caminho_pasta_inmet = os.path.join(pasta_dados, 'inmet_data')
    df_inmet = processa_arquivos_inmet(pasta_inmet=caminho_pasta_inmet)
    if df_inmet.empty:
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
    
    # Preenche valores de enchente onde não houve correspondência
    df_final['CHEIAS_201'] = df_final['CHEIAS_201'].fillna(0)
    df_final.rename(columns={'CHEIAS_201': 'ENCHENTE'}, inplace=True)
    
    print("SUCESSO: Datasets combinados com sucesso!")

    # --- 4. Salvar o dataset final no SQLite ---
    print("Salvando o dataset final no banco de dados...")
    conn = sqlite3.connect('dados_combinados.db')
    df_final.to_sql('dataset_final', conn, if_exists='replace', index=False)
    conn.close()
    
    print("Dataset final salvo em 'dados_combinados.db'.")

    return df_final


if __name__ == '__main__':
    df_final = prepara_datasets_para_treinamento()
    if df_final is not None:
        print("\n--- VISUALIZAÇÃO DO DATASET FINAL ---")
        print(df_final.head())
        print("\nShape do dataset final:", df_final.shape)