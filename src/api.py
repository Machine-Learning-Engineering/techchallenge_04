"""
API RESTFUL PARA MODELO LSTM
Servidor de predições com validação, monitoramento e documentação automática
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from flasgger import Flasgger
from functools import wraps
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import json

from inference_monitoring import LSTMInferenceEngineWithMonitoring
from monitoring_config import MonitoringPresets


# ============================================================================
# Configuração da Aplicação
# ============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Inicializar Swagger/Flasgger
try:
    swagger = Flasgger(app)
except Exception as e:
    logger_init = logging.getLogger("LSTMApi")
    logger_init.warning(f"Erro ao inicializar Flasgger: {e}")

# Carregar spec OpenAPI do arquivo YAML (para expor specs por API)
OPENAPI_SPEC = None
SPEC_FILE = Path(__file__).resolve().parent / "swagger.yml"
if SPEC_FILE.exists():
    try:
        import yaml
        with open(SPEC_FILE, 'r') as f:
            OPENAPI_SPEC = yaml.safe_load(f)
    except ImportError:
        logger_init = logging.getLogger("LSTMApi")
        logger_init.warning("PyYAML nao instalado, swagger.yml sera ignorado")
    except Exception as e:
        logger_init = logging.getLogger("LSTMApi")
        logger_init.warning(f"Erro ao carregar swagger.yml: {e}")

TAG_ALIASES = {
    "all": "all",
    "sistema": "Sistema",
    "predicoes": "Predições",
    "dados": "Dados",
    "analise": "Análise",
    "monitoramento": "Monitoramento"
}

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LSTMApi")

# Configuração global
API_VERSION = "1.0.0"
MODEL_PATH = "lstm_model_AAPL.keras"
DATA_PATH = "data/NASDAQ100_Historical_Data.csv"
MONITORING_DIR = "monitoring_logs"

# Estado da aplicação
inference_engine = None
df_historical = None
app_state = {
    "started_at": datetime.now(),
    "requests_count": 0,
    "errors_count": 0,
    "model_loaded": False
}


# ============================================================================
# Utilities
# ============================================================================

def load_app_state():
    """Carrega estado da aplicação na inicialização"""
    global inference_engine, df_historical
    
    try:
        logger.info("🚀 Inicializando aplicação...")
        
        # Carrega modelo com monitoramento
        logger.info(f"📂 Carregando modelo: {MODEL_PATH}")
        inference_engine = LSTMInferenceEngineWithMonitoring(
            model_path=MODEL_PATH,
            monitoring_dir=MONITORING_DIR
        )
        
        # Carrega dados históricos
        logger.info(f"📂 Carregando dados: {DATA_PATH}")
        df_historical = pd.read_csv(DATA_PATH)
        df_historical['Date'] = pd.to_datetime(df_historical['Date'])
        
        app_state["model_loaded"] = True
        logger.info("✅ Aplicação inicializada com sucesso")
        
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar: {e}")
        app_state["model_loaded"] = False
        raise


def validate_ticker(ticker: str) -> bool:
    """Valida se ticker existe nos dados"""
    if df_historical is None:
        return False
    return ticker in df_historical['Ticker'].unique()


def get_ticker_data(ticker: str, days: int = 365) -> Tuple[np.ndarray, bool]:
    """
    Extrai dados históricos de um ticker
    
    Args:
        ticker: Símbolo do ticker
        days: Número de dias históricos
        
    Returns:
        (prices_array, success)
    """
    if df_historical is None:
        return None, False
    
    df_ticker = df_historical[df_historical['Ticker'] == ticker].copy()
    df_ticker = df_ticker.sort_values('Date').tail(days)
    
    if len(df_ticker) == 0:
        return None, False
    
    prices = df_ticker['Close'].values
    return prices, True


def extract_prices_from_request(data: Dict) -> Tuple[np.ndarray, bool]:
    """
    Extrai array de preços da requisição
    
    Args:
        data: JSON payload
        
    Returns:
        (prices_array, success)
    """
    # Opção 1: Preços fornecidos diretamente
    if 'prices' in data:
        try:
            prices = np.array(data['prices'], dtype=float)
            if len(prices) < 10:
                return None, False
            return prices, True
        except (ValueError, TypeError):
            return None, False
    
    # Opção 2: Ticker + dias
    if 'ticker' in data and 'days' in data:
        ticker = data['ticker'].upper()
        days = int(data['days'])
        
        if not validate_ticker(ticker):
            return None, False
        
        if days < 10 or days > 1000:
            return None, False
        
        return get_ticker_data(ticker, days)
    
    return None, False


def build_openapi_spec_for_tag(tag: str) -> Dict:
    """Filtra o OpenAPI por tag para expor specs separados."""
    if not OPENAPI_SPEC:
        return None

    if not tag or tag == "all":
        return deepcopy(OPENAPI_SPEC)

    spec = deepcopy(OPENAPI_SPEC)
    paths = spec.get("paths", {})
    filtered_paths = {}

    for path, methods in paths.items():
        filtered_methods = {}
        for method, details in methods.items():
            method_tags = details.get("tags", [])
            if tag in method_tags:
                filtered_methods[method] = details
        if filtered_methods:
            filtered_paths[path] = filtered_methods

    spec["paths"] = filtered_paths
    spec["tags"] = [t for t in spec.get("tags", []) if t.get("name") == tag]
    info = spec.get("info", {})
    info["title"] = f"{info.get('title', 'API')} - {tag}"
    spec["info"] = info
    return spec


def track_request(success: bool = True):
    """Decorator para rastrear requisições"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            app_state["requests_count"] += 1
            
            if not success:
                app_state["errors_count"] += 1
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# ============================================================================
# Endpoints
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health Check
    
    Returns:
        Status da aplicação
    """
    return jsonify({
        "status": "healthy" if app_state["model_loaded"] else "unhealthy",
        "version": API_VERSION,
        "model_loaded": app_state["model_loaded"],
        "uptime_seconds": (datetime.now() - app_state["started_at"]).total_seconds(),
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/v1/predict', methods=['POST'])
@track_request()
def predict_single():
    """
    Predição Single-Step
    
    Request (JSON):
    {
        "prices": [263.5, 264.0, 265.2, ...],  // Array com últimos preços
        // OU
        "ticker": "AAPL",                       // Símbolo do ticker
        "days": 60                              // Dias históricos
    }
    
    Response:
    {
        "success": true,
        "prediction": {
            "ticker": "AAPL",
            "current_price": 264.35,
            "predicted_price": 270.95,
            "change": 6.60,
            "change_pct": 2.50,
            "confidence": "HIGH",
            "volatility": 0.015,
            "trend": "UP"
        },
        "metadata": {
            "model": "LSTM",
            "sequence_length": 60,
            "timestamp": "2026-02-26T16:10:00"
        },
        "monitoring": {
            "latency_ms": 45.3
        }
    }
    """
    try:
        # Validação básica
        if not app_state["model_loaded"]:
            return jsonify({
                "success": False,
                "error": "Model not loaded"
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Invalid JSON payload"
            }), 400
        
        # Extrai ticker (se fornecido)
        ticker = data.get("ticker", "N/A").upper()
        
        # Extrai dados
        prices, success = extract_prices_from_request(data)
        if not success:
            return jsonify({
                "success": False,
                "error": "Invalid prices or ticker data"
            }), 400
        
        # Realiza predição
        logger.info(f"Predição single-step com {len(prices)} preços (Ticker: {ticker})")
        prediction = inference_engine.predict_single(prices)
        
        # Calcula tendência
        change_pct = prediction["change_pct"]
        if change_pct > 1.0:
            trend = "🔺 UP"
        elif change_pct < -1.0:
            trend = "🔻 DOWN"
        else:
            trend = "➡️ STABLE"
        
        return jsonify({
            "success": True,
            "prediction": {
                "ticker": ticker,
                "current_price": round(prediction["current_price"], 2),
                "predicted_price": round(prediction["predicted_price"], 2),
                "change": round(prediction["change"], 2),
                "change_pct": round(prediction["change_pct"], 2),
                "confidence": prediction["confidence"],
                "volatility": round(prediction["volatility"], 4),
                "trend": trend
            },
            "metadata": {
                "model": "LSTM",
                "sequence_length": inference_engine.sequence_length,
                "timestamp": datetime.now().isoformat()
            },
            "monitoring": {
                "latency_ms": round(prediction["latency_ms"], 2)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Erro em predict_single: {e}")
        app_state["errors_count"] += 1
        
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/v1/predict-multi', methods=['POST'])
@track_request()
def predict_multiple():
    """
    Predição Multi-Step (Múltiplos dias)
    
    Request (JSON):
    {
        "prices": [263.5, 264.0, ...],  // Array com últimos preços
        // OU
        "ticker": "AAPL",
        "days": 60,
        
        "forecast_days": 15  // Opcional (default: 15, max: 30)
    }
    
    Response:
    {
        "success": true,
        "predictions": [
            {
                "day": 1,
                "predicted_price": 270.95,
                "change": 6.60,
                "change_pct": 2.50
            },
            ...
        ],
        "summary": {
            "total_days": 15,
            "initial_price": 264.35,
            "final_price": 273.23,
            "total_change": 8.88,
            "total_change_pct": 3.36
        },
        "metadata": {
            "model": "LSTM",
            "timestamp": "2026-02-26T16:10:00"
        }
    }
    """
    try:
        if not app_state["model_loaded"]:
            return jsonify({
                "success": False,
                "error": "Model not loaded"
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Invalid JSON payload"
            }), 400
        
        # Extrai dados
        prices, success = extract_prices_from_request(data)
        if not success:
            return jsonify({
                "success": False,
                "error": "Invalid prices or ticker data"
            }), 400
        
        # Número de dias a prever
        forecast_days = data.get('forecast_days', 15)
        forecast_days = max(1, min(30, int(forecast_days)))
        
        # Realiza predição multi-step
        logger.info(f"Predição multi-step: {forecast_days} dias")
        predictions = inference_engine.predict_multiple(prices, days=forecast_days)
        
        # Calcula resumo
        initial_price = prices[-1]
        final_price = predictions[-1]["predicted_price"]
        total_change = final_price - initial_price
        total_change_pct = (total_change / initial_price * 100)
        
        return jsonify({
            "success": True,
            "predictions": [
                {
                    "day": p["day"],
                    "predicted_price": round(p["predicted_price"], 2),
                    "change": round(p["change"], 2),
                    "change_pct": round(p["change_pct"], 2)
                }
                for p in predictions
            ],
            "summary": {
                "total_days": len(predictions),
                "initial_price": round(initial_price, 2),
                "final_price": round(final_price, 2),
                "total_change": round(total_change, 2),
                "total_change_pct": round(total_change_pct, 2),
                "trend": "upward" if total_change > 0 else "downward"
            },
            "metadata": {
                "model": "LSTM",
                "timestamp": datetime.now().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Erro em predict_multiple: {e}")
        app_state["errors_count"] += 1
        
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/v1/tickers', methods=['GET'])
def list_tickers():
    """
    Lista todos os tickers disponíveis
    
    Response:
    {
        "success": true,
        "tickers": ["AAPL", "MSFT", "GOOGL", ...],
        "count": 103
    }
    """
    if df_historical is None:
        return jsonify({
            "success": False,
            "error": "Data not loaded"
        }), 503
    
    tickers = sorted(df_historical['Ticker'].unique().tolist())
    
    return jsonify({
        "success": True,
        "tickers": tickers,
        "count": len(tickers)
    }), 200


@app.route('/api/v1/ticker/<ticker>', methods=['GET'])
def get_ticker_info(ticker: str):
    """
    Informações sobre um ticker específico
    
    Response:
    {
        "success": true,
        "ticker": "AAPL",
        "records_count": 6571,
        "data_range": {
            "from": "2009-01-02",
            "to": "2025-12-31"
        },
        "latest_price": 264.35,
        "price_range": {
            "min": 0.20,
            "max": 285.92,
            "mean": 49.39
        }
    }
    """
    ticker = ticker.upper()
    
    if df_historical is None:
        return jsonify({
            "success": False,
            "error": "Data not loaded"
        }), 503
    
    if not validate_ticker(ticker):
        return jsonify({
            "success": False,
            "error": f"Ticker {ticker} not found"
        }), 404
    
    df_ticker = df_historical[df_historical['Ticker'] == ticker].copy()
    df_ticker = df_ticker.sort_values('Date')
    
    prices = df_ticker['Close'].values
    
    return jsonify({
        "success": True,
        "ticker": ticker,
        "records_count": len(df_ticker),
        "data_range": {
            "from": df_ticker['Date'].min().strftime('%Y-%m-%d'),
            "to": df_ticker['Date'].max().strftime('%Y-%m-%d')
        },
        "latest_price": round(float(prices[-1]), 2),
        "price_range": {
            "min": round(float(np.min(prices)), 2),
            "max": round(float(np.max(prices)), 2),
            "mean": round(float(np.mean(prices)), 2)
        }
    }), 200


@app.route('/api/v1/monitoring/report', methods=['GET'])
def get_monitoring_report():
    """
    Relatório de monitoramento da API
    
    Response:
    {
        "success": true,
        "report": {
            "uptime_seconds": 3600,
            "requests_total": 1234,
            "errors_total": 2,
            "latency": {...},
            "resources": {...},
            "health": {...}
        }
    }
    """
    if inference_engine is None:
        return jsonify({
            "success": False,
            "error": "Monitoring not available"
        }), 503
    
    try:
        report = inference_engine.get_monitoring_report()
        
        return jsonify({
            "success": True,
            "report": {
                "api": {
                    "uptime_seconds": report.get("uptime_seconds", 0),
                    "requests_total": app_state["requests_count"],
                    "errors_total": app_state["errors_count"],
                    "error_rate": (
                        app_state["errors_count"] / max(1, app_state["requests_count"]) * 100
                    )
                },
                "inference": report.get("latency", {}),
                "resources": report.get("resources", {}),
                "health": report.get("health_status", {})
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Erro ao gerar relatório: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/v1/info', methods=['GET'])
def api_info():
    """
    Informações da API
    
    Response:
    {
        "success": true,
        "info": {
            "name": "LSTM Stock Price Prediction API",
            "version": "1.0.0",
            "model": "LSTM",
            "model_loaded": true,
            "endpoints": [...]
        }
    }
    """
    endpoints = [
        {
            "method": "GET",
            "path": "/health",
            "description": "Health check"
        },
        {
            "method": "POST",
            "path": "/api/v1/predict",
            "description": "Single-step prediction"
        },
        {
            "method": "POST",
            "path": "/api/v1/predict-multi",
            "description": "Multi-step prediction"
        },
        {
            "method": "GET",
            "path": "/api/v1/tickers",
            "description": "List available tickers"
        },
        {
            "method": "GET",
            "path": "/api/v1/ticker/<ticker>",
            "description": "Get ticker information"
        },
        {
            "method": "GET",
            "path": "/api/v1/monitoring/report",
            "description": "Get monitoring report"
        },
        {
            "method": "GET",
            "path": "/api/v1/analytics/<ticker>",
            "description": "Get technical analysis"
        },
        {
            "method": "GET",
            "path": "/api/v1/logs",
            "description": "Get system logs"
        },
        {
            "method": "GET",
            "path": "/api/v1/info",
            "description": "API information"
        }
    ]
    
    return jsonify({
        "success": True,
        "info": {
            "name": "LSTM Stock Price Prediction API",
            "version": API_VERSION,
            "description": "RESTful API for LSTM-based stock price forecasting",
            "documentation": {
                "swagger_ui": "http://localhost:5001/swagger-ui/",
                "swagger_index": "http://localhost:5001/apidocs/",
                "openapi_full": "http://localhost:5001/openapi.json"
            },
            "model": {
                "type": "LSTM",
                "sequence_length": 60 if inference_engine else None,
                "loaded": app_state["model_loaded"],
                "parameters": inference_engine.model.count_params() if inference_engine else None
            },
            "endpoints": endpoints
        }
    }), 200


@app.route('/openapi.json', methods=['GET'])
def openapi_full():
    """Retorna a especificacao OpenAPI completa."""
    if not OPENAPI_SPEC:
        return jsonify({
            "success": False,
            "error": "OpenAPI spec not available"
        }), 503
    return jsonify(OPENAPI_SPEC), 200


@app.route('/swagger/<tag>.json', methods=['GET'])
def openapi_by_tag(tag: str):
    """Retorna a especificacao OpenAPI filtrada por tag."""
    if not OPENAPI_SPEC:
        return jsonify({
            "success": False,
            "error": "OpenAPI spec not available"
        }), 503

    tag_key = tag.lower()
    if tag_key not in TAG_ALIASES:
        return jsonify({
            "success": False,
            "error": "Tag not found",
            "available": sorted(TAG_ALIASES.keys())
        }), 404

    resolved_tag = TAG_ALIASES[tag_key]
    if resolved_tag == "all":
        return jsonify(OPENAPI_SPEC), 200

    spec = build_openapi_spec_for_tag(resolved_tag)
    if not spec:
        return jsonify({
            "success": False,
            "error": "OpenAPI spec not available"
        }), 503

    return jsonify(spec), 200


def apidocs_index():
    """Pagina com links para os Swagger specs por API."""
    links = [
        {"label": "Todas as APIs", "url": "http://localhost:5001/openapi.json"},
        {"label": "Sistema", "url": "http://localhost:5001/swagger/sistema.json"},
        {"label": "Predicoes", "url": "http://localhost:5001/swagger/predicoes.json"},
        {"label": "Dados", "url": "http://localhost:5001/swagger/dados.json"},
        {"label": "Analise", "url": "http://localhost:5001/swagger/analise.json"},
        {"label": "Monitoramento", "url": "http://localhost:5001/swagger/monitoramento.json"}
    ]

    html = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Swagger Links - LSTM API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f7f9fc; }
            h1 { color: #2f4b9a; }
            ul { list-style: none; padding: 0; }
            li { margin: 12px 0; }
            a { text-decoration: none; color: #1a73e8; font-weight: 600; }
            a:hover { text-decoration: underline; }
            .note { color: #555; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>Links Swagger por API</h1>
        <p>Escolha a especificacao que deseja abrir:</p>
        <ul>
            {% for link in links %}
            <li><a href="{{ link.url }}" target="_blank">{{ link.label }}</a></li>
            {% endfor %}
        </ul>
        <p class="note">Swagger UI: <a href="http://localhost:5001/swagger-ui/" target="_blank">/swagger-ui/</a></p>
    </body>
    </html>
    """

    return render_template_string(html, links=links), 200


@app.route('/api-docs', methods=['GET'])
def api_docs_page():
    """Pagina com enderecos das APIs e documentacao."""
    endpoints = [
        {"method": "GET", "path": "/health", "desc": "Health check"},
        {"method": "POST", "path": "/api/v1/predict", "desc": "Single-step prediction"},
        {"method": "POST", "path": "/api/v1/predict-multi", "desc": "Multi-step prediction"},
        {"method": "GET", "path": "/api/v1/tickers", "desc": "List available tickers"},
        {"method": "GET", "path": "/api/v1/ticker/<ticker>", "desc": "Get ticker information"},
        {"method": "GET", "path": "/api/v1/analytics/<ticker>", "desc": "Get technical analysis"},
        {"method": "GET", "path": "/api/v1/monitoring/report", "desc": "Get monitoring report"},
        {"method": "GET", "path": "/api/v1/logs", "desc": "Get system logs"},
        {"method": "GET", "path": "/api/v1/info", "desc": "API information"}
    ]

    docs = [
        {"label": "Swagger UI", "url": "http://localhost:5001/swagger-ui/"},
        {"label": "OpenAPI (todas as APIs)", "url": "http://localhost:5001/openapi.json"},
        {"label": "Swagger Sistema", "url": "http://localhost:5001/swagger/sistema.json"},
        {"label": "Swagger Predicoes", "url": "http://localhost:5001/swagger/predicoes.json"},
        {"label": "Swagger Dados", "url": "http://localhost:5001/swagger/dados.json"},
        {"label": "Swagger Analise", "url": "http://localhost:5001/swagger/analise.json"},
        {"label": "Swagger Monitoramento", "url": "http://localhost:5001/swagger/monitoramento.json"}
    ]

    html = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>APIs e Documentacao - LSTM</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f7f9fc; color: #1f2a44; }
            h1 { color: #2f4b9a; margin-bottom: 10px; }
            h2 { margin-top: 30px; color: #2f4b9a; }
            .section { background: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 6px 18px rgba(0,0,0,0.08); }
            ul { list-style: none; padding: 0; }
            li { margin: 10px 0; }
            a { text-decoration: none; color: #1a73e8; font-weight: 600; }
            a:hover { text-decoration: underline; }
            .endpoint { display: flex; gap: 12px; align-items: center; }
            .method { min-width: 60px; font-weight: 700; color: #34495e; }
            .path { font-family: "Courier New", monospace; background: #eef2ff; padding: 4px 8px; border-radius: 6px; }
            .desc { color: #5a677a; }
        </style>
    </head>
    <body>
        <h1>Enderecos das APIs e Documentacao</h1>
        <div class="section">
            <h2>Documentacao</h2>
            <ul>
                {% for item in docs %}
                <li><a href="{{ item.url }}" target="_blank">{{ item.label }}</a></li>
                {% endfor %}
            </ul>
        </div>

        <div class="section" style="margin-top: 20px;">
            <h2>Endpoints</h2>
            <ul>
                {% for e in endpoints %}
                <li class="endpoint">
                    <span class="method">{{ e.method }}</span>
                    <span class="path">{{ e.path }}</span>
                    <span class="desc">{{ e.desc }}</span>
                </li>
                {% endfor %}
            </ul>
        </div>
    </body>
    </html>
    """

    return render_template_string(html, endpoints=endpoints, docs=docs), 200


# Registrar /apidocs apenas se nao houver conflito com Flasgger
if not any(rule.rule == "/apidocs/" for rule in app.url_map.iter_rules()):
    app.add_url_rule("/apidocs/", "apidocs_index", apidocs_index)


def swagger_ui_page():
    """Swagger UI customizado apontando para os specs da API."""
    urls = [
        {"url": "/openapi.json", "name": "All APIs"},
        {"url": "/swagger/sistema.json", "name": "Sistema"},
        {"url": "/swagger/predicoes.json", "name": "Predicoes"},
        {"url": "/swagger/dados.json", "name": "Dados"},
        {"url": "/swagger/analise.json", "name": "Analise"},
        {"url": "/swagger/monitoramento.json", "name": "Monitoramento"}
    ]

    html = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Swagger UI - LSTM API</title>
        <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
        <style>
            body { margin: 0; background: #f4f6fb; }
            .topbar { display: none; }
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
        <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
        <script>
            window.ui = SwaggerUIBundle({
                urls: {{ urls | tojson }},
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                layout: "StandaloneLayout"
            });
        </script>
    </body>
    </html>
    """

    return render_template_string(html, urls=urls), 200


# Registrar /swagger-ui apenas se nao houver conflito
if not any(rule.rule == "/swagger-ui/" for rule in app.url_map.iter_rules()):
    app.add_url_rule("/swagger-ui/", "swagger_ui_page", swagger_ui_page)


@app.route('/api/v1/analytics/<ticker>', methods=['GET'])
def get_ticker_analytics(ticker: str):
    """
    Análise técnica de um ticker
    
    Query params:
        - period: número de dias (default: 90)
    
    Response:
    {
        "success": true,
        "analytics": {
            "ticker": "AAPL",
            "dates": ["2025-11-01", ...],
            "prices": [263.5, 264.2, ...],
            "volatility": [0.015, 0.018, ...],
            "volume": [50000000, ...]
        }
    }
    """
    ticker = ticker.upper()
    period = int(request.args.get('period', 90))
    
    if df_historical is None:
        return jsonify({
            "success": False,
            "error": "Data not loaded"
        }), 503
    
    if not validate_ticker(ticker):
        return jsonify({
            "success": False,
            "error": f"Ticker {ticker} not found"
        }), 404
    
    try:
        df_ticker = df_historical[df_historical['Ticker'] == ticker].copy()
        df_ticker = df_ticker.sort_values('Date')
        df_ticker = df_ticker.tail(period)
        
        # Calcular volatilidade
        returns = df_ticker['Close'].pct_change().dropna()
        volatility = returns.rolling(window=20).std().tolist()
        
        # Preparar dados
        dates = [d.strftime('%Y-%m-%d') for d in df_ticker['Date']]
        prices = df_ticker['Close'].tolist()
        volumes = df_ticker['Volume'].tolist() if 'Volume' in df_ticker.columns else [0] * len(prices)
        
        return jsonify({
            "success": True,
            "analytics": {
                "ticker": ticker,
                "period": period,
                "dates": dates,
                "prices": prices,
                "volumes": volumes,
                "volatility": [float(v) if not pd.isna(v) else 0 for v in volatility],
                "returns": returns.tolist()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Erro na análise: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/v1/logs', methods=['GET'])
def get_logs():
    """
    Retorna logs do sistema
    
    Query params:
        - limit: número de logs (default: 100)
        - type: tipo de log (prediction, inference, drift, error, etc)
    
    Response:
    {
        "success": true,
        "logs": [
            {
                "timestamp": "2026-02-26T16:10:00",
                "type": "prediction",
                "message": "Predição realizada com sucesso"
            },
            ...
        ]
    }
    """
    limit = int(request.args.get('limit', 100))
    log_type = request.args.get('type', '')
    
    try:
        # Tentar ler logs de eventos
        logs_list = []
        metrics_list = []
        
        monitoring_dir = Path(MONITORING_DIR)
        
        # Ler eventos
        if (monitoring_dir / "events.jsonl").exists():
            with open(monitoring_dir / "events.jsonl", 'r') as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        if not log_type or event.get('event_type') == log_type:
                            logs_list.append({
                                'timestamp': event.get('timestamp', ''),
                                'type': event.get('event_type', 'unknown'),
                                'message': event.get('message', '')
                            })
        
        # Ler métricas
        if (monitoring_dir / "metrics.jsonl").exists():
            with open(monitoring_dir / "metrics.jsonl", 'r') as f:
                for line in f:
                    if line.strip():
                        metric = json.loads(line)
                        metrics_list.append({
                            'timestamp': metric.get('timestamp', ''),
                            'type': metric.get('metric', ''),
                            'metric': metric.get('metric', ''),
                            'value': metric.get('value', 0)
                        })
        
        # Retornar últimos N logs
        logs_list = sorted(logs_list, key=lambda x: x['timestamp'], reverse=True)[:limit]
        metrics_list = sorted(metrics_list, key=lambda x: x['timestamp'], reverse=True)[:limit]
        
        return jsonify({
            "success": True,
            "logs": logs_list,
            "metrics": metrics_list,
            "count": len(logs_list)
        }), 200
        
    except Exception as e:
        logger.error(f"Erro ao retornar logs: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "success": False,
        "error": "Method not allowed"
    }), 405


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500


# ============================================================================
# Startup/Shutdown
# ============================================================================

def shutdown():
    """Executado ao finalizar a aplicação"""
    logger.info("Aplicação finalizada")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    try:
        # Carrega estado
        load_app_state()
        
        # Inicia servidor
        print("\n" + "="*70)
        print("🚀 LSTM Stock Price Prediction API")
        print("="*70)
        print(f"📚 Swagger UI: http://localhost:5001/apidocs")
        print(f"📋 Swagger JSON: http://localhost:5001/swagger.json")
        print(f"📖 API Info: http://localhost:5001/api/v1/info")
        print(f"❤️  Health Check: http://localhost:5001/health")
        print("="*70)
        print(f"\n✅ Modelo carregado: {app_state['model_loaded']}")
        print(f"   Sequência: 60 dias | Parâmetros: ~{inference_engine.model.count_params() if inference_engine else '?'} ")
        print("="*70 + "\n")
        
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=False,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        exit(1)
