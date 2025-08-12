import pandas as pd
import os
import sqlite3

def consolidar_dados(pasta_inmet, arquivo_inundacao, arquivo_vulnerabilidade):
    """
    Consolida os arquivos CSV em um único DataFrame e o salva em um banco de dados SQLite.
    """
    
    # === 1. Ler, limpar e consolidar os arquivos CSV do INMET ===
    print("Iniciando a leitura e consolidação dos arquivos do INMET...")
    
    lista_dfs_inmet = []
    caminho_pasta_inmet = os.path.join(os.path.dirname(__file__), 'dados', pasta_inmet)
    
    # Mapeamento dos nomes de colunas originais para nomes simples e unificados
    mapeamento_flexivel = {
        # Mapeamentos para a coluna 'Precipitacao'
        'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'Precipitacao',
        'CHUVA': 'Precipitacao',
        'CHUVA(mm)': 'Precipitacao',
        
        # Mapeamentos para a coluna 'Umidade'
        'UMIDADE RELATIVA DO AR, HORARIA (%)': 'Umidade',
        'UMIDADE RELATIVA DO AR (%)': 'Umidade',
        
        # Mapeamentos para a coluna 'Vento'
        'VENTO, VELOCIDADE HORARIA (m/s)': 'Vento',
        'VELOCIDADE DO VENTO, HORARIA (m/s)': 'Vento',

        # Mapeamentos para a coluna 'Temperatura'
        'TEMPERATURA DO AR - BULBO SECO, HORARIA (\xc2\xb0C)': 'Temperatura',
        'TEMPERATURA DO AR (\xc2\xb0C)': 'Temperatura',
        'TEMPERATURA DO AR (°C)': 'Temperatura',
        'TEMP_AR': 'Temperatura',
        'TEMPERATURA': 'Temperatura',
        
        # Mapeamentos para a coluna 'Data'
        'DATA (YYYY-MM-DD)': 'Data',
        'Data': 'Data',

        # Mapeamentos para a coluna 'Hora'
        'HORA (UTC)': 'Hora',
        'Hora': 'Hora'
    }
    
    # Colunas que queremos no DataFrame final
    colunas_finais = ['DataHora', 'Precipitacao', 'Umidade', 'Vento', 'Temperatura']
    
    for arquivo in os.listdir(caminho_pasta_inmet):
        if arquivo.endswith('.CSV'):
            caminho_arquivo = os.path.join(caminho_pasta_inmet, arquivo)
            try:
                # O INMET usa delimitador ';' e ponto para decimal
                df = pd.read_csv(caminho_arquivo, sep=';', decimal=',', encoding='latin1', skiprows=8)
                
                # Normaliza os nomes das colunas (remove espaços extras)
                df.columns = df.columns.str.strip()
                
                # Renomeia as colunas do DataFrame usando o mapeamento flexível
                colunas_para_renomear = {}
                for nome_origem, nome_destino in mapeamento_flexivel.items():
                    if nome_origem in df.columns and nome_destino not in df.columns:
                        colunas_para_renomear[nome_origem] = nome_destino
                
                df = df.rename(columns=colunas_para_renomear)
                
                # Verifica quais colunas estão presentes no arquivo e as renomeia para o formato final
                colunas_presentes = [c for c in colunas_finais if c in df.columns]
                
                # Se 'Data' e 'Hora' estiverem presentes, unifica-as em 'DataHora'
                if 'Data' in df.columns and 'Hora' in df.columns:
                    df['DataHora'] = pd.to_datetime(df['Data'] + ' ' + df['Hora'], format='%Y-%m-%d %H:%M')
                    colunas_presentes.append('DataHora')
                
                # Seleciona apenas as colunas de interesse
                df_limpo = df[colunas_presentes].copy()
                
                # Garante que todas as colunas finais existam, preenchendo as faltantes com NaN
                for col in colunas_finais:
                    if col not in df_limpo.columns:
                        df_limpo[col] = pd.NA
                        
                lista_dfs_inmet.append(df_limpo)
                print(f"SUCESSO: Arquivo {arquivo} processado.")

            except Exception as e:
                print(f"ERRO: Falha ao ler ou processar o arquivo {arquivo}: {e}")
                
    if not lista_dfs_inmet:
        print("Nenhum arquivo do INMET foi lido. Verifique o caminho e o formato dos arquivos.")
        return
        
    df_inmet = pd.concat(lista_dfs_inmet, ignore_index=True)
    
    # === 2. Ler e processar os arquivos CSV da ANA ===
    print("Lendo arquivos da ANA...")
    try:
        caminho_ana_inundacao = os.path.join(os.path.dirname(__file__), 'dados', arquivo_inundacao)
        caminho_ana_vulnerabilidade = os.path.join(os.path.dirname(__file__), 'dados', arquivo_vulnerabilidade)

        df_inundacao = pd.read_csv(caminho_ana_inundacao, sep=';', encoding='latin1')
        df_vulnerabilidade = pd.read_csv(caminho_ana_vulnerabilidade, sep=';', encoding='latin1')
    except Exception as e:
        print(f"Erro ao ler arquivos da ANA. Verifique o caminho e o formato. Erro: {e}")
        return
        
    df_inundacao['Enchente'] = 1
    
    
    # === 3. Unir os datasets ===
    print("Unindo datasets...")
    print("AVISO: A união entre os dados do INMET e da ANA não foi possível com as colunas atuais.")
    print("O script vai continuar, mas o dataset final não terá informações de inundação.")
    df_final = df_inmet.copy()
    
    
    # === 4. Salvar o dataset final no SQLite ===
    print("Salvando o dataset final no banco de dados...")
    conn = sqlite3.connect("database.db")
    df_final.to_sql('clima_historico', conn, if_exists='replace', index=False)
    conn.close()
    
    print("Processo de consolidação concluído! Os dados estão na tabela 'clima_historico' no database.db.")

if __name__ == '__main__':
    pasta_do_inmet = 'inmet_data'
    arquivo_da_ana_inundacao = 'ana_inundacao.csv'
    arquivo_da_ana_vulnerabilidade = 'ana_vulnerabilidade.csv'
    
    consolidar_dados(pasta_do_inmet, arquivo_da_ana_inundacao, arquivo_da_ana_vulnerabilidade)