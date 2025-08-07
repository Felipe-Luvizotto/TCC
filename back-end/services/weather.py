import requests, os, time

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

def get_weather_data(lat, lon, tentativas=3):
    """
    Busca dados meteorológicos (temperatura, umidade, vento, precipitação)
    para uma coordenada geográfica.
    """
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={lat},{lon}"
    for _ in range(tentativas):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Extraindo a precipitação em milímetros (mm)
            precipitacao_mm = data["current"].get("precip_mm", 0)
            
            # A API retorna 'precip_mm' para a precipitação.
            # O .get() é usado para evitar erros caso a chave não exista no retorno da API.
            
            return [data["current"]["temp_c"], data["current"]["humidity"], data["current"]["wind_kph"], precipitacao_mm]
        except Exception:
            # Em caso de erro, espera e tenta novamente.
            time.sleep(2)
    return None