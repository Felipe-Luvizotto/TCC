# --- START OF FILE prepara_dados.py ---
import pandas as pd
import glob
import os
import sqlite3
import unidecode
import logging
from database import criar_tabelas # Importar a função para criar as tabelas

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
            # Lê o arquivo duas vezes: uma para o cabeçalho e outra para os dados
            # Isso é para lidar com o formato específico do INMET onde o cabeçalho está na linha 9 (índice 8)
            df_header_raw = pd.read_csv(arquivo, sep=';', encoding='latin1', skiprows=8, nrows=1, header=None)
            colunas = df_header_raw.iloc[0].values

            df_data = pd.read_csv(arquivo, sep=';', encoding='latin1', skiprows=9, header=None, names=colunas, on_bad_lines='skip')
            
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
                logging.warning(f"AVISO: Colunas ausentes no arquivo {os.path.basename(arquivo)}: {colunas_ausentes}. Colunas esperadas: {colunas_finais}. Colunas no DF: {df_data.columns.tolist()}")
                continue
            
            df_data = df_data[colunas_finais]

            # 4. Adiciona o código da estação
            df_data['CODIGOESTACAO'] = codigo_estacao
            
            # 5. Converte os dados (exceto Data e Hora que serão tratadas depois)
            for col in ['Precipitacao', 'Temperatura', 'Umidade', 'Vento', 'Pressao', 'Radiacao']:
                df_data[col] = pd.to_numeric(df_data[col].astype(str).str.replace(',', '.'), errors='coerce')
            
            # 6. Filtra valores inválidos (-9999)
            df_data = df_data.replace(-9999, np.nan) # Substitui por NaN para tratar com dropna

            df_data['CODIGOESTACAO'] = df_data['CODIGOESTACAO'].astype(str)
            
            lista_dfs_inmet.append(df_data)
        
        except Exception as e:
            logging.error(f"ERRO ao processar o arquivo {os.path.basename(arquivo)}: {e}")
            continue

    if not lista_dfs_inmet:
        logging.warning("Nenhum DataFrame válido do INMET foi gerado. Retornando DataFrame vazio.")
        return pd.DataFrame()
        
    df_inmet = pd.concat(lista_dfs_inmet, ignore_index=True)
    # Garante que as colunas essenciais para o modelo não sejam NaN
    df_inmet.dropna(subset=['Precipitacao', 'Temperatura', 'Umidade', 'Vento'], inplace=True)
    
    logging.info(f"CONCLUÍDO: {len(arquivos_inmet)} arquivos processados. Total de linhas: {len(df_inmet)}.")
    return df_inmet

def prepara_e_salva_dados():
    """
    Orquestra o processo de preparação e salvamento dos dados no banco de dados.
    """
    criar_tabelas() # Garante que as tabelas existem

    pasta_dados = os.path.join(os.path.dirname(__file__), 'dados')

    # Carregar base de dados de inundações da ANA
    logging.info("Processando base de dados de inundações da ANA...")
    caminho_inundacao = os.path.join(pasta_dados, 'ana_inundacao.csv')
    df_inundacao = pd.read_csv(caminho_inundacao)
    df_inundacao['NM_MUNICIP_NORMALIZADO'] = df_inundacao['NM_MUNICIP'].apply(normalizar_nome_municipio)
    logging.info(f"Base da ANA carregada com {len(df_inundacao)} linhas.")

    # Carregar catálogo de estações do INMET
    logging.info("Criando mapeamento de estações INMET para códigos IBGE...")
    caminho_catalogo = os.path.join(pasta_dados, 'catalogoestacoesautomaticas.csv')
    df_catalogo = pd.read_csv(caminho_catalogo, delimiter=';', encoding='latin1', decimal=',')
    df_catalogo['DC_NOME_NORMALIZADO'] = df_catalogo['DC_NOME'].apply(normalizar_nome_municipio)

    df_catalogo['CD_ESTACAO'] = df_catalogo['CD_ESTACAO'].astype(str)
    
    # Popular a tabela 'municipios'
    conn = sqlite3.connect('database.db')
    df_municipios_para_db = df_catalogo[['DC_NOME', 'VL_LATITUDE', 'VL_LONGITUDE']].rename(columns={'DC_NOME': 'nome', 'VL_LATITUDE': 'latitude', 'VL_LONGITUDE': 'longitude'})
    df_municipios_para_db.drop_duplicates(subset=['nome'], inplace=True)
    try:
        df_municipios_para_db.to_sql('municipios', conn, if_exists='replace', index=False)
        logging.info(f"Tabela 'municipios' populada com {len(df_municipios_para_db)} entradas.")
    except Exception as e:
        logging.error(f"ERRO ao popular a tabela 'municipios': {e}")
    conn.close()


    # Mesclar para criar o mapeamento de estações INMET para IBGE e nome de município
    mapeamento_estacao_municipio = pd.merge(
        df_catalogo[['CD_ESTACAO', 'DC_NOME', 'DC_NOME_NORMALIZADO', 'VL_LATITUDE', 'VL_LONGITUDE']],
        df_inundacao[['CD_GEOCMU', 'NM_MUNICIP_NORMALIZADO', 'NM_MUNICIP']].drop_duplicates(),
        left_on='DC_NOME_NORMALIZADO',
        right_on='NM_MUNICIP_NORMALIZADO',
        how='inner'
    )
    logging.info(f"Mapeamento de estação para código IBGE criado com {len(mapeamento_estacao_municipio)} linhas.")
    
    if mapeamento_estacao_municipio.empty:
        logging.warning("Mapeamento entre estações INMET e municípios ANA resultou em um DataFrame vazio. Verifique os nomes normalizados.")
        logging.warning(f"Exemplo de nomes INMET normalizados: {df_catalogo['DC_NOME_NORMALIZADO'].head().tolist()}")
        logging.warning(f"Exemplo de nomes ANA normalizados: {df_inundacao['NM_MUNICIP_NORMALIZADO'].head().tolist()}")


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
        how='inner' # Mudado para inner para garantir correspondência
    )
    logging.info(f"INMET e mapeamento mesclados. Total de linhas: {len(df_inmet_com_ibge)}.")
    
    if df_inmet_com_ibge.empty:
        logging.warning("INMET e mapeamento resultaram em DataFrame vazio. Verifique 'CODIGOESTACAO' no INMET e 'CD_ESTACAO' no catálogo.")
        logging.warning(f"Exemplo de CODIGOESTACAO (INMET): {df_inmet['CODIGOESTACAO'].unique()[:5].tolist()}")
        logging.warning(f"Exemplo de CD_ESTACAO (Catálogo): {mapeamento_estacao_municipio['CD_ESTACAO'].unique()[:5].tolist()}")
        return None


    # Mesclar com dados de inundações (apenas a coluna 'CHEIAS_201' para a label 'Enchente')
    # Usamos o 'NM_MUNICIP_NORMALIZADO' do mapeamento para garantir a correspondência correta com a tabela de inundações
    df_final = pd.merge(
        df_inmet_com_ibge,
        df_inundacao[['NM_MUNICIP_NORMALIZADO', 'CHEIAS_201']].drop_duplicates(), # Usar a coluna normalizada para o merge
        on='NM_MUNICIP_NORMALIZADO',
        how='left'
    )

    df_final.rename(columns={'CHEIAS_201': 'Enchente', 'NM_MUNICIP': 'municipio'}, inplace=True)
    df_final['Enchente'] = df_final['Enchente'].fillna(0).astype(int)
    
    # Use 'DC_NOME' do catálogo para o nome final do município, pois é mais confiável para as estações.
    # Já está em df_inmet_com_ibge
    df_final.rename(columns={'DC_NOME': 'municipio'}, inplace=True)

    # Filtra linhas onde 'municipio' é nulo (onde o merge não encontrou correspondência final)
    df_final.dropna(subset=['municipio'], inplace=True)
    
    # Remove colunas desnecessárias antes de salvar na tabela 'clima'
    # Mantenha apenas o que é relevante para o modelo e a identificação do registro.
    colunas_para_salvar_clima = [
        'Data', 'Hora', 'Precipitacao', 'Temperatura', 'Umidade', 'Vento', 'Pressao', 'Radiacao', 'Enchente', 'municipio'
    ]
    df_final = df_final[colunas_para_salvar_clima]

    logging.info(f"SUCESSO: Datasets combinados com sucesso! Dataset final para 'clima' com {len(df_final)} linhas.")

    # Conexão com o banco de dados
    conn = sqlite3.connect('database.db')
    
    logging.info("Salvando o dataset final no banco de dados 'clima'...")
    # Limpa a tabela antes de inserir para evitar duplicatas em cada execução
    conn.execute("DELETE FROM clima")
    df_final.to_sql('clima', conn, if_exists='append', index=False)
    
    logging.info("Dados salvos com sucesso na tabela 'clima'!")
    
    conn.close()

if __name__ == '__main__':
    prepara_e_salva_dados()