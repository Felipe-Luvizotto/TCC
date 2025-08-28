import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker } from 'react-leaflet';
import MarkerClusterGroup from 'react-leaflet-markercluster';
import 'leaflet/dist/leaflet.css';
import 'leaflet.markercluster/dist/MarkerCluster.css';
import 'leaflet.markercluster/dist/MarkerCluster.Default.css';
import './MapaBrasil.css'
import axios from 'axios';
import L from 'leaflet';
import { Link } from 'react-router-dom';
import { Bar, Line } from 'react-chartjs-2';
import { Chart, registerables } from 'chart.js';
import { Spinner } from 'react-bootstrap'; // Se você estiver usando react-bootstrap para o spinner
import 'bootstrap/dist/css/bootstrap.min.css'; // Importe o CSS do Bootstrap

Chart.register(...registerables);

// Novo URL para o endpoint de estações
const ESTACOES_URL = 'http://localhost:8000/estacoes/';

const API_URL = 'http://localhost:8000/predict/';
const EVALUATE_URL = 'http://localhost:8000/evaluate/';
const HISTORY_API_URL = 'http://localhost:8000/predict/history/';

const defaultIcon = new L.Icon({
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
});

const redIcon = new L.Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
});

const yellowIcon = new L.Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-yellow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
});

const greenIcon = new L.Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
});

const MapaBrasil = () => {
  const [dados, setDados] = useState({});
  const [municipioSelecionado, setMunicipioSelecionado] = useState(null);
  const [evaluationMetrics, setEvaluationMetrics] = useState(null);
  const [evaluationChartData, setEvaluationChartData] = useState(null);
  const [timeSeriesChartData, setTimeSeriesChartData] = useState(null);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(true);

  // Alteração no useEffect para buscar os dados do novo endpoint
  useEffect(() => {
    // Busca dados das estações do backend
    axios.get(ESTACOES_URL)
      .then(response => {
        const estacoesData = response.data.estacoes;
        const dadosFormatados = {};
        estacoesData.forEach(estacao => {
          dadosFormatados[estacao.nome] = estacao;
        });
        setDados(dadosFormatados);
        setLoading(false);
      })
      .catch(error => {
        console.error("Erro ao carregar dados das estações:", error);
        setLoading(false);
      });
      
    // Busca as métricas de avaliação
    axios.get(EVALUATE_URL)
      .then(response => {
        setEvaluationMetrics(response.data);
      })
      .catch(error => {
        console.error("Erro ao carregar métricas de avaliação:", error);
      });
  }, []);

  useEffect(() => {
    if (evaluationMetrics) {
      const labels = ['Acurácia', 'Precisão', 'Recall', 'F1-Score'];
      const data = {
        labels: labels,
        datasets: [
          {
            label: 'Modelo Ensemble',
            data: [
              evaluationMetrics.Ensemble.accuracy,
              evaluationMetrics.Ensemble.precision,
              evaluationMetrics.Ensemble.recall,
              evaluationMetrics.Ensemble.f1_score,
            ],
            backgroundColor: 'rgba(75, 192, 192, 0.6)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
          },
          {
            label: 'Random Forest',
            data: [
              evaluationMetrics.Random_Forest.accuracy,
              evaluationMetrics.Random_Forest.precision,
              evaluationMetrics.Random_Forest.recall,
              evaluationMetrics.Random_Forest.f1_score,
            ],
            backgroundColor: 'rgba(153, 102, 255, 0.6)',
            borderColor: 'rgba(153, 102, 255, 1)',
            borderWidth: 1,
          },
          {
            label: 'XGBoost',
            data: [
              evaluationMetrics.XGBoost.accuracy,
              evaluationMetrics.XGBoost.precision,
              evaluationMetrics.XGBoost.recall,
              evaluationMetrics.XGBoost.f1_score,
            ],
            backgroundColor: 'rgba(255, 159, 64, 0.6)',
            borderColor: 'rgba(255, 159, 64, 1)',
            borderWidth: 1,
          },
          {
            label: 'LSTM',
            data: [
              evaluationMetrics.LSTM.accuracy,
              evaluationMetrics.LSTM.precision,
              evaluationMetrics.LSTM.recall,
              evaluationMetrics.LSTM.f1_score,
            ],
            backgroundColor: 'rgba(54, 162, 235, 0.6)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1,
          }
        ],
      };
      setEvaluationChartData(data);
    }
  }, [evaluationMetrics]);
  
  const getIcon = (probabilidade) => {
    if (probabilidade >= 0.75) {
      return redIcon;
    } else if (probabilidade >= 0.5) {
      return yellowIcon;
    } else {
      return greenIcon;
    }
  };

  const predict = async (municipio) => {
    setLoadingPrediction(true);
    setPredictionResult(null);
    setMunicipioSelecionado(municipio); // Define o município selecionado

    try {
      const response = await axios.post(API_URL, {
        municipio: municipio.nome,
      });
      setPredictionResult(response.data);
      console.log('Dados de previsão:', response.data);

      // Chamar a API de histórico se a previsão for bem-sucedida
      fetchTimeSeriesData(municipio.nome);

    } catch (error) {
      console.error('Erro na previsão:', error);
      if (error.response && error.response.data && error.response.data.detail) {
        setPredictionResult({ error: error.response.data.detail });
      } else {
        setPredictionResult({ error: 'Erro ao conectar com a API de previsão.' });
      }
    } finally {
      setLoadingPrediction(false);
    }
  };

  const fetchTimeSeriesData = async (municipioNome) => {
    try {
      const response = await axios.post(HISTORY_API_URL, {
        municipio: municipioNome,
      });

      if (response.data.noData) {
        setTimeSeriesChartData({ noData: true });
        return;
      }

      const history = response.data.history;

      const labels = history.map(item => item.data_hora);
      const dataPoints = history.map(item => item.probabilidade_enchente);

      setTimeSeriesChartData({
        labels: labels,
        datasets: [
          {
            label: 'Probabilidade de Enchente',
            data: dataPoints,
            fill: false,
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1,
          },
        ],
      });
    } catch (error) {
      console.error('Erro ao buscar dados históricos:', error);
      setTimeSeriesChartData({ erro: true });
    }
  };
  
  const handleCloseSidebar = () => {
    setMunicipioSelecionado(null);
    setPredictionResult(null);
  };

  const evaluationChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
      },
    },
  };

  const timeSeriesChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
        title: {
          display: true,
          text: 'Probabilidade',
        },
      },
      x: {
        title: {
          display: true,
          text: 'Data e Hora',
        },
      },
    },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              label += `${(context.parsed.y * 100).toFixed(2)}%`;
            }
            return label;
          }
        }
      }
    }
  };

  if (loading) {
    return (
      <div className="d-flex justify-content-center align-items-center" style={{ height: '100vh' }}>
        <Spinner animation="border" role="status">
          <span className="sr-only">Carregando mapa e estações...</span>
        </Spinner>
      </div>
    );
  }

  return (
    <>
      <Link to="/dicas" className="dicas-link">Dicas 🚨</Link>
      <MapContainer center={[-15.7797, -47.9297]} zoom={4} style={{ height: '100vh', width: '100vw' }}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        <MarkerClusterGroup>
          {Object.values(dados).map((municipio, index) => (
            <Marker 
              key={index} 
              position={[municipio.lat, municipio.lon]} 
              icon={defaultIcon}
              eventHandlers={{
                click: () => predict(municipio),
              }}
            />
          ))}
        </MarkerClusterGroup>
      </MapContainer>
      
      {municipioSelecionado && (
        <div className="sidebar">
          <button className="sidebar-close" onClick={handleCloseSidebar}>&times;</button>
          <h2>{municipioSelecionado.nome}</h2>
          {loadingPrediction ? (
            <div className="text-center">
              <Spinner animation="border" size="sm" />
              <p>Carregando previsão...</p>
            </div>
          ) : predictionResult && predictionResult.error ? (
            <p style={{ color: 'red' }}>{predictionResult.error}</p>
          ) : predictionResult ? (
            <>
              <p>A probabilidade de enchente hoje é: <strong>{(predictionResult.probabilidade_enchente * 100).toFixed(2)}%</strong></p>
              <div className="prediction-details">
                <p>Temperatura: {predictionResult.dados_atuais.Temperatura}°C</p>
                <p>Umidade: {predictionResult.dados_atuais.Umidade}%</p>
                <p>Vento: {predictionResult.dados_atuais.Vento} km/h</p>
                <p>Precipitação: {predictionResult.dados_atuais.Precipitacao} mm</p>
              </div>

              <h3>Histórico de Probabilidade</h3>
              <div style={{ height: '250px' }}>
                {timeSeriesChartData?.erro ? (
                  <p style={{ color: 'red' }}>Erro ao carregar histórico.</p>
                ) : timeSeriesChartData?.noData ? (
                  <p>Não há dados históricos para este município.</p>
                ) : timeSeriesChartData ? (
                  <Line data={timeSeriesChartData} options={timeSeriesChartOptions} />
                ) : (
                  <p>Não há dados históricos disponíveis ou ocorreu um erro.</p>
                )}
              </div>
            </>
          ) : null }
        </div>
      )}

      <div className="evaluation-panel">
        <h3>Avaliação do Modelo Ensemble</h3>
        {evaluationMetrics === null ? (
          <p>Carregando métricas de avaliação...</p>
        ) : evaluationMetrics.erro ? (
          <p style={{ color: 'red' }}>Erro ao carregar métricas.</p>
        ) : (
          <>
            {evaluationChartData && (
              <div style={{ height: '250px' }}>
                <Bar data={evaluationChartData} options={evaluationChartOptions} />
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
};

export default MapaBrasil;