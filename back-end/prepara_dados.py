import pandas as pd
import glob
import os
import sqlite3
import unidecode
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    logging.info("Iniciando a leitura e consolidação dos arquivos do INMET...")

    lista_dfs_inmet = []
    
    arquivos_inmet = glob.glob(os.path.join(pasta_inmet, 'INMET_*.CSV')) + \
                     glob.glob(os.path.join(pasta_inmet, 'INMET_*.csv'))
    
    if not arquivos_inmet:
        logging.warning("AVISO: Nenhum arquivo INMET encontrado na pasta.")
        return pd.DataFrame()

    for arquivo in arquivos_inmet:
        try:
            logging.info(f"Processando arquivo: {os.path.basename(arquivo)}")

            # 1. Extrai o código da estação do cabeçalho do arquivo
            codigo_estacao = None
            with open(arquivo, 'r', encoding='latin1') as f:
                for linha in f:
                    if 'CODIGO (WMO):;' in linha:
                        codigo_estacao = linha.split(';')[1].strip()
                        break
            
            if codigo_estacao is None:
                logging.warning(f"AVISO: Código de estação não encontrado no arquivo {os.path.basename(arquivo)}. Arquivo será ignorado.")
                continue

            # 2. Encontra a linha de cabeçalho (nomes das colunas)
            df_temp = pd.read_csv(arquivo, sep=';', encoding='latin1', skiprows=8, header=None, on_bad_lines='skip')
            
            # Pega os nomes das colunas da nona linha (índice 8)
            colunas = df_temp.iloc[0].values
            df_data = df_temp.iloc[1:].copy()
            df_data.columns = colunas
            df_data.reset_index(drop=True, inplace=True)
            
            # 3. Limpeza e seleção de colunas relevantes
            colunas_selecionadas = {
                'DATA (YYYY-MM-DD)': 'Data',
                'HORA (UTC)': 'Hora',
                'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'Precipitacao',
                'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'Temperatura',
                'UMIDADE RELATIVA DO AR, HORARIA (%)': 'Umidade',
                'VENTO, VELOCIDADE HORARIA (m/s)': 'Vento',
                'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': 'Pressao',
                'RADIACAO GLOBAL (KJ/m²)':'Radiacao',
            }
            df_data = df_data.rename(columns=colunas_selecionadas)
            
            colunas_finais = list(colunas_selecionadas.values())
            
            # Verificar se as colunas necessárias estão presentes
            colunas_ausentes = [c for c in colunas_finais if c not in df_data.columns]
            if colunas_ausentes:
                logging.warning(f"AVISO: Colunas ausentes no arquivo {os.path.basename(arquivo)}: {colunas_ausentes}")
                continue
            
            df_data = df_data[colunas_finais]

            # 4. Adiciona o código da estação
            df_data['CODIGOESTACAO'] = codigo_estacao
            
            # 5. Converte os dados
            df_data = df_data.apply(pd.to_numeric, errors='coerce')
            
            # 6. Filtra valores inválidos (-9999)
            df_data = df_data[df_data != -9999]

            # --- CORREÇÃO: Converte 'CODIGOESTACAO' para string antes de adicionar à lista ---
            df_data['CODIGOESTACAO'] = df_data['CODIGOESTACAO'].astype(str)
            
            lista_dfs_inmet.append(df_data)
        
        except Exception as e:
            logging.error(f"ERRO ao processar o arquivo {os.path.basename(arquivo)}: {e}")
            continue

    if not lista_dfs_inmet:
        return pd.DataFrame()
        
    df_inmet = pd.concat(lista_dfs_inmet, ignore_index=True)
    df_inmet.dropna(subset=['Precipitacao', 'Temperatura', 'Umidade', 'Vento'], inplace=True)
    
    logging.info(f"CONCLUÍDO: {len(arquivos_inmet)} arquivos processados. Total de linhas: {len(df_inmet)}.")
    return df_inmet

def prepara_e_salva_dados():
    """
    Orquestra o processo de preparação e salvamento dos dados no banco de dados.
    """
    pasta_dados = os.path.join(os.path.dirname(__file__), 'dados')

    # Carregar base de dados de inundações da ANA
    logging.info("Processando base de dados de inundações da ANA...")
    caminho_inundacao = os.path.join(pasta_dados, 'ana_inundacao.csv')
    df_inundacao = pd.read_csv(caminho_inundacao)
    df_inundacao['NM_MUNICIP_NORMALIZADO'] = df_inundacao['NM_MUNICIP'].apply(normalizar_nome_municipio)
    logging.info("Processamento da base da ANA concluído.")

    # Carregar catálogo de estações do INMET
    logging.info("Criando mapeamento de estações INMET para códigos IBGE...")
    caminho_catalogo = os.path.join(pasta_dados, 'catalogoestacoesautomaticas.csv')
    df_catalogo = pd.read_csv(caminho_catalogo, delimiter=';', encoding='latin1')
    df_catalogo['DC_NOME_NORMALIZADO'] = df_catalogo['DC_NOME'].apply(normalizar_nome_municipio)

    # --- CORREÇÃO: Converte 'CD_ESTACAO' para string antes de usar na mesclagem ---
    df_catalogo['CD_ESTACAO'] = df_catalogo['CD_ESTACAO'].astype(str)

    # Mesclar para criar o mapeamento
    mapeamento_estacao_municipio = pd.merge(
        df_catalogo[['CD_ESTACAO', 'DC_NOME_NORMALIZADO']],
        df_inundacao[['CD_GEOCMU', 'NM_MUNICIP_NORMALIZADO', 'NM_MUNICIP']].drop_duplicates(),
        left_on='DC_NOME_NORMALIZADO',
        right_on='NM_MUNICIP_NORMALIZADO',
        how='inner'
    )
    logging.info("Mapeamento de código IBGE criado.")

    # Processar os arquivos INMET
    caminho_pasta_inmet = os.path.join(pasta_dados, 'inmet_data')
    df_inmet = processa_arquivos_inmet(pasta_inmet=caminho_pasta_inmet)
    if df_inmet.empty:
        logging.warning("Finalizando o script: nenhum dado do INMET processado.")
        return None

    # Mesclar dados do INMET com o mapeamento
    df_inmet_com_ibge = pd.merge(
        df_inmet,
        mapeamento_estacao_municipio,
        left_on='CODIGOESTACAO',
        right_on='CD_ESTACAO',
        how='left'
    )
    
    # Mesclar com dados de inundações
    df_final = pd.merge(
        df_inmet_com_ibge,
        df_inundacao[['CD_GEOCMU', 'CHEIAS_201']].drop_duplicates(),
        on='CD_GEOCMU',
        how='left'
    )

    df_final.rename(columns={'CHEIAS_201': 'Enchente'}, inplace=True)
    df_final['Enchente'] = df_final['Enchente'].fillna(0).astype(int)
    
    # Adicionar o nome do município (original)
    df_final = pd.merge(
        df_final,
        df_inundacao[['CD_GEOCMU', 'NM_MUNICIP']].drop_duplicates(),
        on='CD_GEOCMU',
        how='left'
    )
    df_final.rename(columns={'NM_MUNICIP': 'municipio'}, inplace=True)
    
    # Filtra linhas onde 'CD_GEOCMU' é nulo (onde o merge não encontrou correspondência)
    df_final.dropna(subset=['CD_GEOCMU'], inplace=True)
    
    # Remove colunas desnecessárias antes de salvar
    df_final.drop(columns=['DC_NOME_NORMALIZADO', 'CD_ESTACAO', 'CD_GEOCMU', 'NM_MUNICIP_NORMALIZADO', 'CODIGOESTACAO'], inplace=True, errors='ignore')

    # Sucesso, agora df_final está pronto para ser salvo.
    logging.info(f"SUCESSO: Datasets combinados com sucesso! Dataset final com {len(df_final)} linhas.")

    # Conexão com o banco de dados
    conn = sqlite3.connect('database.db')
    
    logging.info("Salvando o dataset final no banco de dados...")
    
    # Salvar o DataFrame no banco de dados. Use 'append' se a tabela já existir.
    df_final.to_sql('clima', conn, if_exists='append', index=False)
    
    logging.info("Dados salvos com sucesso!")
    
    conn.close()

if __name__ == '__main__':
    prepara_e_salva_dados()