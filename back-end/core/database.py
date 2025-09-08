# --- START OF FILE database.py ---
import sqlite3

def criar_tabelas():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Tabela 'clima' - ajustada para refletir as colunas que você está gerando em prepara_dados.py
    # Adicionamos 'Radiacao'
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clima (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Data TEXT NOT NULL,
            Hora TEXT NOT NULL,
            Precipitacao REAL,
            Temperatura REAL,
            Umidade REAL,
            Vento REAL,
            Pressao REAL,
            Radiacao REAL,
            Enchente INTEGER,
            municipio TEXT NOT NULL
        );
    """)

    # Tabela 'municipios' - para armazenar as coordenadas e nomes dos municípios/estações
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS municipios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL UNIQUE,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL
        );
    """)
    
    # Tabela 'historico_previsao' - para armazenar o histórico de previsões
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historico_previsao (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            municipio TEXT NOT NULL,
            data_hora TEXT NOT NULL,
            probabilidade REAL NOT NULL
        );
    """)

    conn.commit()
    conn.close()

# Remove as funções inserir_dados_clima, pois o pandas fará isso com to_sql

if __name__ == '__main__':
    criar_tabelas()