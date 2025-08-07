import sqlite3

def criar_tabela_clima():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Adicione a coluna 'precipitacao'
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clima (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            municipio TEXT NOT NULL,
            data_hora TEXT NOT NULL,
            temperatura REAL,
            umidade REAL,
            vento REAL,
            precipitacao REAL,
            enchente INTEGER
        );
    """)

    conn.commit()
    conn.close()

def inserir_dados_clima(municipio, data_hora, temperatura, umidade, vento, precipitacao, enchente):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO clima (municipio, data_hora, temperatura, umidade, vento, precipitacao, enchente)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (municipio, data_hora, temperatura, umidade, vento, precipitacao, enchente))

    conn.commit()
    conn.close()