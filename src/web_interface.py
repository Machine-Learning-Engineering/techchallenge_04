#!/usr/bin/env python3
"""
Web Interface - Dashboard principal para LSTM Stock Price Prediction
Porta: 8080
"""

from flask import Flask, render_template_string, jsonify
from flask_cors import CORS
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# URLs dos serviços
SERVICES = {
    'api': 'http://localhost:5001',
    'dashboard': 'http://localhost:5000',
    'interface': 'http://localhost:8080'
}

# HTML da página principal
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LSTM Stock Price Prediction - Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            padding: 40px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
        }
        
        .header h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 1.1em;
            margin-bottom: 15px;
        }
        
        .status-badges {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .badge {
            display: inline-block;
            padding: 8px 16px;
            background: #f0f0f0;
            border-radius: 25px;
            font-size: 0.9em;
            font-weight: 600;
            color: #333;
            border-left: 4px solid #667eea;
        }
        
        .badge.online {
            background: #e8f5e9;
            border-left-color: #4caf50;
            color: #2e7d32;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.2);
        }
        
        .card h2 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .card p {
            color: #666;
            line-height: 1.6;
            margin-bottom: 20px;
        }
        
        .btn {
            display: inline-block;
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            font-size: 1em;
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-secondary {
            background: #f0f0f0;
            color: #333;
            border: 2px solid #667eea;
        }
        
        .btn-secondary:hover {
            background: #667eea;
            color: white;
        }
        
        .card-links {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .btn-small {
            padding: 8px 16px;
            font-size: 0.9em;
        }
        
        .footer {
            background: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            color: #666;
            margin-top: 30px;
        }
        
        /* Modal */
        #predictionModal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
            display: none;
            justify-content: center;
            align-items: center;
        }
        
        .modal-content {
            background: white;
            padding: 40px;
            border-radius: 15px;
            width: 90%;
            max-width: 500px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        
        .modal-content h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.3em;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #333;
        }
        
        .form-group input,
        .form-group select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
        }
        
        .modal-buttons {
            display: flex;
            gap: 10px;
            margin-top: 25px;
        }
        
        .modal-buttons button {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            font-size: 1em;
        }
        
        .btn-predict {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-cancel {
            background: #f0f0f0;
            color: #333;
        }
        
        #predictionResult {
            margin-top: 20px;
            padding: 20px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e8ecf1 100%);
            border-radius: 12px;
            border-left: 4px solid #667eea;
            display: none;
        }
        
        #predictionResult h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        
        #resultContent {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .result-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .result-label {
            font-size: 0.85em;
            color: #999;
            margin-bottom: 5px;
        }
        
        .result-value {
            font-size: 1.3em;
            font-weight: bold;
        }
        
        .result-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9em;
            color: white;
        }
        
        .loading {
            text-align: center;
            color: #667eea;
            font-weight: 600;
            display: none;
            margin-top: 20px;
        }
        
        #logsModal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
            display: none;
            justify-content: center;
            align-items: center;
        }
        
        #logsContent {
            background: white;
        }
        
        .log-line {
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .log-line:last-child {
            border-bottom: none;
        }
        
        .log-time {
            color: #667eea;
            font-weight: bold;
        }
        
        .log-type {
            font-weight: 600;
            margin: 0 5px;
        }
        
        .log-info {
            color: #2196F3;
        }
        
        .log-error {
            color: #f44336;
        }
        
        .log-warning {
            color: #ff9800;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🚀 LSTM Stock Price Prediction</h1>
            <p>Previsões de preços de ações usando redes neurais de aprendizado profundo</p>
            <div class="status-badges">
                <span class="badge online">✓ API Operacional</span>
                <span class="badge">v1.0.0</span>
                <span class="badge">Atualizado: <span id="time"></span></span>
            </div>
        </div>
        
        <!-- Card Principal -->
        <div class="grid">
            <div class="card">
                <h2><span>🔮</span> Testar Predição</h2>
                <p>Faça predições de preços de ações usando redes neurais LSTM.</p>
                <div class="card-links">
                    <button class="btn btn-secondary btn-small" onclick="window.showPredictionModal()">
                        Testar Predição
                    </button>
                </div>
            </div>
            
            <div class="card">
                <h2><span>📋</span> Logs</h2>
                <p>Visualize os registros de eventos e operações do sistema.</p>
                <div class="card-links">
                    <button class="btn btn-secondary btn-small" onclick="window.showLogsModal()">
                        Ver Logs
                    </button>
                </div>
            </div>
            
            <div class="card">
                <h2><span>📊</span> Monitoramento</h2>
                <p>Acesse o dashboard de monitoramento em tempo real do sistema.</p>
                <div class="card-links">
                    <a href="#" onclick="window.open(DASHBOARD_URL, '_blank'); return false;" class="btn btn-secondary btn-small">
                        Ir para Monitoramento
                    </a>
                </div>
            </div>
            
            <div class="card">
                <h2><span>📚</span> Documentação API</h2>
                <p>Explore a documentação interativa e testes dos endpoints da API.</p>
                <div class="card-links">
                    <a href="#" onclick="window.open(API_URL + '/api-docs', '_blank'); return false;" class="btn btn-secondary btn-small">
                        Swagger UI
                    </a>
                    <a href="#" onclick="window.open(API_URL + '/api-docs', '_blank'); return false;" class="btn btn-secondary btn-small">
                        API Info
                    </a>
                </div>
            </div>
        </div>
        
        <!-- Modal de Predição -->
        <div id="predictionModal">
            <div class="modal-content">
                <h2>🔮 Testar Predição</h2>
                
                <div class="form-group">
                    <label for="tickerSelect">Ticker:</label>
                    <select id="tickerSelect">
                        <option value="">Carregando...</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="historyDays">Dias Históricos:</label>
                    <input type="number" id="historyDays" value="60" min="10" max="1000">
                </div>
                
                <div id="predictionLoading" class="loading">
                    ⏳ Processando predição...
                </div>
                
                <div id="predictionResult">
                    <h3>✨ Resultado da Predição</h3>
                    <div id="resultContent"></div>
                </div>
                
                <div class="modal-buttons">
                    <button class="btn-predict" onclick="window.makePrediction()">
                        Fazer Predição
                    </button>
                    <button class="btn-cancel" onclick="window.closePredictionModal()">
                        Cancelar
                    </button>
                </div>
            </div>
        </div>
        
        <!-- Modal de Logs -->
        <div id="logsModal">
            <div class="modal-content">
                <h2>📋 Logs do Sistema</h2>
                
                <div id="logsLoading" class="loading">
                    ⏳ Carregando logs...
                </div>
                
                <div id="logsContent" style="display: none; max-height: 400px; overflow-y: auto; background: #f8f9fa; border-radius: 8px; padding: 15px; margin-bottom: 15px; font-family: 'Courier New', monospace; font-size: 0.85em; line-height: 1.6;">
                </div>
                
                <div class="modal-buttons">
                    <button class="btn" onclick="window.loadLogs()" style="background: #667eea;">
                        Recarregar Logs
                    </button>
                    <button class="btn-cancel" onclick="window.closeLogsModal()">
                        Fechar
                    </button>
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>LSTM Stock Price Prediction v1.0 | Última atualização: <span id="time2"></span></p>
        </div>
    </div>

    <script>
        // Detectar dinamicamente as URLs dos serviços baseado no hostname do navegador
        const API_HOST = window.location.hostname;
        const API_URL = `http://${API_HOST}:5001`;
        const DASHBOARD_URL = `http://${API_HOST}:5000`;
        
        console.log('Detectado API_URL:', API_URL);
        console.log('Detectado DASHBOARD_URL:', DASHBOARD_URL);
        
        // Funções globais para o window
        window.showPredictionModal = function() {
            console.log('showPredictionModal chamado');
            document.getElementById('predictionModal').style.display = 'flex';
            loadTickers();
        };
        
        window.closePredictionModal = function() {
            console.log('closePredictionModal chamado');
            document.getElementById('predictionModal').style.display = 'none';
            document.getElementById('predictionResult').style.display = 'none';
            document.getElementById('predictionLoading').classList.remove('loading');
        };
        
        window.makePrediction = function() {
            console.log('makePrediction chamado');
            const ticker = document.getElementById('tickerSelect').value;
            const days = document.getElementById('historyDays').value;
            
            if (!ticker) {
                alert('Por favor, selecione um ticker');
                return;
            }
            
            const loading = document.getElementById('predictionLoading');
            const result = document.getElementById('predictionResult');
            
            loading.style.display = 'block';
            result.style.display = 'none';
            
            fetch(`${API_URL}/api/v1/predict`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ticker: ticker, days: parseInt(days)})
            })
            .then(response => response.json())
            .then(data => {
                console.log('Resposta recebida:', data);
                loading.style.display = 'none';
                result.style.display = 'block';
                
                if (data.success) {
                    displayPredictionResult(data.prediction, data.monitoring.latency_ms);
                } else {
                    showPredictionError(data.error || 'Erro desconhecido');
                }
            })
            .catch(error => {
                console.error('Erro:', error);
                loading.style.display = 'none';
                result.style.display = 'block';
                showPredictionError('Erro na requisição: ' + error.message);
            });
        };
        
        function loadTickers() {
            console.log('loadTickers chamado');
            fetch(`${API_URL}/api/v1/tickers`)
                .then(r => r.json())
                .then(data => {
                    const select = document.getElementById('tickerSelect');
                    select.innerHTML = '<option value="">Selecione um ticker...</option>';
                    
                    if (data.tickers && data.tickers.length > 0) {
                        data.tickers.slice(0, 50).forEach(ticker => {
                            const option = document.createElement('option');
                            option.value = ticker;
                            option.textContent = ticker;
                            select.appendChild(option);
                        });
                    }
                })
                .catch(e => {
                    console.error('Erro ao carregar tickers:', e);
                    document.getElementById('tickerSelect').innerHTML = '<option value="">Erro ao carregar tickers</option>';
                });
        }
        
        function displayPredictionResult(pred, latency) {
            const container = document.getElementById('resultContent');
            container.innerHTML = '';
            
            // Determinar cores
            let trendColor = '#667eea';
            if (pred.trend && pred.trend.includes('UP')) trendColor = '#4caf50';
            else if (pred.trend && pred.trend.includes('DOWN')) trendColor = '#f44336';
            
            let confColor = '#2196F3';
            if (pred.confidence === 'HIGH') confColor = '#4caf50';
            else if (pred.confidence === 'MEDIUM') confColor = '#ff9800';
            else if (pred.confidence === 'LOW') confColor = '#ff5722';
            
            // Adicionar cards
            addResultCard(container, 'TICKER', pred.ticker, '#667eea');
            addResultCard(container, 'PREÇO ATUAL', '$' + pred.current_price.toFixed(2), '#333');
            addResultCard(container, 'PREÇO PREDITO', '$' + pred.predicted_price.toFixed(2), '#667eea');
            addResultCard(container, 'TENDÊNCIA', pred.trend, trendColor);
            
            const changeColor = pred.change_pct > 0 ? '#4caf50' : '#f44336';
            addResultCard(container, 'MUDANÇA', '$' + pred.change.toFixed(2) + ' (' + pred.change_pct.toFixed(2) + '%)', changeColor);
            
            addResultCardBadge(container, 'CONFIANÇA', pred.confidence, confColor);
            addResultCard(container, 'VOLATILIDADE', pred.volatility.toFixed(4), '#2196F3');
            addResultCard(container, 'LATÊNCIA', Math.round(latency) + ' ms', '#666');
        }
        
        function addResultCard(container, label, value, color) {
            const card = document.createElement('div');
            card.className = 'result-card';
            
            const labelDiv = document.createElement('div');
            labelDiv.className = 'result-label';
            labelDiv.textContent = label;
            
            const valueDiv = document.createElement('div');
            valueDiv.className = 'result-value';
            valueDiv.style.color = color;
            valueDiv.textContent = value;
            
            card.appendChild(labelDiv);
            card.appendChild(valueDiv);
            container.appendChild(card);
        }
        
        function addResultCardBadge(container, label, value, color) {
            const card = document.createElement('div');
            card.className = 'result-card';
            
            const labelDiv = document.createElement('div');
            labelDiv.className = 'result-label';
            labelDiv.textContent = label;
            
            const badge = document.createElement('div');
            badge.className = 'result-badge';
            badge.style.background = color;
            badge.textContent = value;
            
            card.appendChild(labelDiv);
            card.appendChild(badge);
            container.appendChild(card);
        }
        
        function showPredictionError(message) {
            const container = document.getElementById('resultContent');
            container.innerHTML = '<div style="grid-column: 1 / -1; color: #f44336; font-weight: bold; padding: 15px; background: #ffebee; border-radius: 8px;">❌ ' + message + '</div>';
        }
        
        // Funções para Logs
        window.showLogsModal = function() {
            console.log('showLogsModal chamado');
            document.getElementById('logsModal').style.display = 'flex';
            window.loadLogs();
        };
        
        window.closeLogsModal = function() {
            console.log('closeLogsModal chamado');
            document.getElementById('logsModal').style.display = 'none';
        };
        
        window.loadLogs = function() {
            console.log('loadLogs chamado');
            const loading = document.getElementById('logsLoading');
            const content = document.getElementById('logsContent');
            
            loading.style.display = 'block';
            content.style.display = 'none';
            
            fetch(`${API_URL}/api/v1/logs?limit=50`)
                .then(response => response.json())
                .then(data => {
                    loading.style.display = 'none';
                    content.style.display = 'block';
                    
                    if (data.logs && data.logs.length > 0) {
                        let logsHtml = '';
                        data.logs.forEach((log, index) => {
                            const time = new Date(log.timestamp).toLocaleString('pt-BR');
                            const typeClass = 'log-' + (log.type || 'info').toLowerCase();
                            logsHtml += '<div class="log-line"><span class="log-time">[' + time + ']</span> <span class="log-type ' + typeClass + '">' + (log.type || 'INFO') + ':</span> <span>' + (log.message || 'N/A') + '</span></div>';
                        });
                        content.innerHTML = logsHtml;
                    } else {
                        content.innerHTML = '<div style="padding: 15px; text-align: center; color: #999;">Nenhum log disponível</div>';
                    }
                })
                .catch(error => {
                    console.error('Erro ao carregar logs:', error);
                    loading.style.display = 'none';
                    content.style.display = 'block';
                    content.innerHTML = '<div style="padding: 15px; color: #f44336;">❌ Erro ao carregar logs: ' + error.message + '</div>';
                });
        };
        
        // Atualizar hora
        function updateTime() {
            const time = new Date().toLocaleString('pt-BR');
            const el1 = document.getElementById('time');
            const el2 = document.getElementById('time2');
            if (el1) el1.textContent = time;
            if (el2) el2.textContent = time;
        }
        
        updateTime();
        setInterval(updateTime, 1000);
        
        console.log('Script carregado com sucesso!');
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Página principal"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'web_interface',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/services')
def services():
    """Listar serviços disponíveis"""
    return jsonify({
        'services': SERVICES,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Iniciando Web Interface na porta 8080...")
    logger.info("Acesse: http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)
