import pandas as pd
import os
import unidecode

def normalizar_nome_municipio(nome):
    """Normaliza o nome do município para um formato consistente."""
    if isinstance(nome, str):
        return unidecode.unidecode(nome).upper().strip()
    return None

def diagnosticar_correspondencia_municipios():
    """
    Verifica a correspondência entre os municípios da base do INMET e da ANA.
    """
    pasta_dados = os.path.join(os.path.dirname(__file__), 'dados')
    
    # Carrega o catálogo de estações do INMET
    caminho_catalogo = os.path.join(pasta_dados, 'catalogoestacoesautomaticas.csv')
    df_catalogo = pd.read_csv(caminho_catalogo, delimiter=';')
    
    # Carrega a base de dados de inundações da ANA
    caminho_inundacao = os.path.join(pasta_dados, 'ana_inundacao.csv')
    df_inundacao = pd.read_csv(caminho_inundacao)
    
    # Normaliza os nomes dos municípios para garantir a correspondência
    df_catalogo['DC_NOME_NORMALIZADO'] = df_catalogo['DC_NOME'].apply(normalizar_nome_municipio)
    df_inundacao['NM_MUNICIP_NORMALIZADO'] = df_inundacao['NM_MUNICIP'].apply(normalizar_nome_municipio)
    
    # Encontra a interseção dos nomes de municípios
    municipios_catalogo = set(df_catalogo['DC_NOME_NORMALIZADO'].unique())
    municipios_inundacao = set(df_inundacao['NM_MUNICIP_NORMALIZADO'].unique())
    
    municipios_em_comum = municipios_catalogo.intersection(municipios_inundacao)
    
    print("--- DIAGNÓSTICO DE CORRESPONDÊNCIA DE MUNICÍPIOS ---")
    print(f"Total de municípios no Catálogo do INMET: {len(municipios_catalogo)}")
    print(f"Total de municípios na base da ANA: {len(municipios_inundacao)}")
    print("-" * 50)
    print(f"Número de municípios em comum: {len(municipios_em_comum)}")
    
    if municipios_em_comum:
        print("Municípios que correspondem nas duas bases:")
        for municipio in sorted(list(municipios_em_comum)):
            print(f"- {municipio}")
    else:
        print("Nenhum município corresponde entre as bases. A união vai falhar.")

if __name__ == '__main__':
    diagnosticar_correspondencia_municipios()