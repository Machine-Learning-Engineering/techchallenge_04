# 🚀 LSTM Stock Price Prediction

> Sistema completo de previsão de preços de ações usando redes neurais LSTM, com API RESTful, interface web interativa e dashboard de monitoramento em tempo real.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Sumário

- [Visão Geral](#-visão-geral)
- [Características](#-características)
- [Arquitetura](#-arquitetura)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Execução com Podman](#-execução-com-podman)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Visão Geral

O **LSTM Stock Price Prediction** é um sistema de machine learning para previsão de preços de ações utilizando redes neurais LSTM (Long Short-Term Memory). O projeto oferece:

- **API RESTful** completa para predições single-step e multi-step
- **Interface Web** intuitiva para testes interativos
- **Dashboard de Monitoramento** com métricas em tempo real
- **Documentação Swagger** interativa para todas as APIs
- **Sistema de Logs** estruturado (eventos e métricas)
- **Containerização** completa com Podman/Docker

O modelo foi treinado com dados históricos do NASDAQ-100 e é capaz de:
- Prever preços futuros de ações
- Calcular tendências (alta, baixa, estável)
- Estimar níveis de confiança
- Analisar volatilidade

---

## ✨ Características

### 🔮 Predições Avançadas
- **Single-step**: Previsão do próximo dia
- **Multi-step**: Previsões para até 30 dias
- **Análise técnica**: Volatilidade, tendências e confiança
- **100+ tickers**: Dados históricos do NASDAQ-100

### 🌐 Interface Completa
- **Dashboard Web**: Interface amigável em português
- **Swagger UI**: Documentação interativa das APIs
- **Visualizações**: Gráficos e cards informativos
- **Responsivo**: Funciona em desktop e mobile

### 📊 Monitoramento Robusto
- **Métricas em tempo real**: Latência, throughput, recursos
- **Detecção de drift**: Alerta sobre degradação do modelo
- **Health checks**: Verificação automática de saúde
- **Logs estruturados**: JSONL para análise posterior

### 🐳 Deploy Simplificado
- **Containerfile otimizado**: Build rápido e eficiente
- **Multi-stage**: Imagem leve (~800MB)
- **Usuário non-root**: Segurança reforçada
- **Health checks**: Restart automático em falhas

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser / Cliente                        │
└─────────────────┬───────────────────────────┬───────────────┘
                  │                           │
         ┌────────▼────────┐         ┌────────▼─────────┐
         │  Web Interface  │         │   Swagger UI     │
         │   (Port 8080)   │         │  /api-docs       │
         └────────┬────────┘         └────────┬─────────┘
                  │                           │
         ┌────────▼───────────────────────────▼─────────┐
         │           API RESTful (Port 5001)            │
         │  ┌──────────────────────────────────────┐    │
         │  │  /api/v1/predict                     │    │
         │  │  /api/v1/predict-multi               │    │
         │  │  /api/v1/tickers                     │    │
         │  │  /api/v1/analytics/<ticker>          │    │
         │  └──────────────────────────────────────┘    │
         └────────┬─────────────────────────────────────┘
                  │
         ┌────────▼──────────┐
         │   LSTM Model      │
         │   (TensorFlow)    │
         │   52,033 params   │
         └────────┬──────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼────┐  ┌─────▼──────┐  ┌──▼────────┐
│ Logs   │  │ Monitoring │  │ Historical│
│ System │  │ Dashboard  │  │   Data    │
│        │  │(Port 5000) │  │  (CSV)    │
└────────┘  └────────────┘  └───────────┘
```

**Componentes:**

1. **Web Interface (8080)**: Dashboard principal com interface amigável
2. **API RESTful (5001)**: Endpoints para predições e dados
3. **Monitoring Dashboard (5000)**: Métricas e health checks
4. **LSTM Model**: Rede neural com 60 timesteps de sequência
5. **Logs System**: Eventos (JSONL) e métricas estruturadas

---

## � Estrutura do Projeto

### Árvore Completa

```
techchallenge_04/
│
├── README.md                              # Documentação completa
├── Containerfile                          # Build da imagem container
├── .gitignore                             # Regras para versionamento Git
│
└── src/                                   # Código-fonte da aplicação
    │
    ├── 🔌 API & Interface
    │   ├── api.py                         # API RESTful (Flask)
    │   │   ├── Endpoints: /api/v1/predict, /api/v1/predict-multi
    │   │   ├── Endpoints: /api/v1/tickers, /api/v1/analytics
    │   │   ├── Swagger UI: /swagger-ui/, /api-docs
    │   │   └── Documentação: /openapi.json, /swagger/<tag>.json
    │   │
    │   └── web_interface.py               # Dashboard Web (Flask)
    │       ├── Principal: http://localhost:8080
    │       ├── 4 Cards: Predição, Logs, Monitoramento, Docs
    │       └── Interface amigável em português
    │
    ├── 🤖 Machine Learning & Modelo
    │   ├── lstm_model.py                  # Definição do modelo LSTM
    │   │   ├── Arquitetura: 2 camadas LSTM + Dense
    │   │   ├── Parâmetros: 52,033
    │   │   ├── Input: Sequência de 60 dias
    │   │   └── Output: Previsão do próximo dia
    │   │
    │   ├── lstm_inference.py              # Engine de inferência
    │   │   ├── Carregamento do modelo
    │   │   ├── Normalização de dados
    │   │   ├── Predição single/multi-step
    │   │   └── Cálculo de métricas (volatilidade, tendência)
    │   │
    │   └── lstm_model_AAPL.keras          # Modelo pré-treinado
    │       └── Treinado com NASDAQ-100 (653KB)
    │
    ├── 📊 Monitoramento & Métricas
    │   ├── monitoring_dashboard.py        # Dashboard de métricas (Flask)
    │   │   ├── Porta: 5000
    │   │   ├── Status do modelo
    │   │   ├── Latência de inferência
    │   │   ├── Recursos (CPU, memória, disco)
    │   │   └── Detecção de drift
    │   │
    │   ├── model_monitoring.py            # Monitor do modelo
    │   │   ├── Detecção de degradação
    │   │   ├── Drift detection
    │   │   ├── Health checks
    │   │   └── Alertas
    │   │
    │   ├── inference_monitoring.py        # Monitor de inferência
    │   │   ├── Latência (média, min, max, p95)
    │   │   ├── Throughput
    │   │   ├── Taxa de erro
    │   │   └── Registro de eventos
    │   │
    │   └── monitoring_config.py           # Configuração do sistema
    │       ├── Thresholds de alerta
    │       ├── Período de monitoramento
    │       ├── Tamanho de janela
    │       └── Limites de recursos
    │
    ├── ⚙️ Configuração & Deploy
    │   ├── requirements.txt               # Dependências Python
    │   │   ├── tensorflow 2.13+
    │   │   ├── flask 2.3+
    │   │   ├── pandas 1.5+
    │   │   ├── numpy 1.24+
    │   │   ├── flasgger 0.9.7+
    │   │   └── (mais 10+ pacotes)
    │   │
    │   ├── swagger.yml                    # Especificação OpenAPI 3.0
    │   │   ├── 9 endpoints documentados
    │   │   ├── 5 categorias (Sistema, Predições, Dados, Análise, Monitoramento)
    │   │   ├── Schemas de request/response
    │   │   └── Exemplos de uso
    │   │
    │   ├── START_ALL.sh                   # Script de inicialização
    │   │   ├── Cria diretórios necessários
    │   │   ├── Inicia API (5001)
    │   │   ├── Inicia Web Interface (8080)
    │   │   ├── Inicia Monitoring Dashboard (5000)
    │   │   └── Aguarda sinais de encerramento
    │   │
    │   ├── run_all.sh                     # Orquestrador de serviços
    │   │   ├── Verifica Python environment
    │   │   ├── Instala dependências
    │   │   ├── Executa serviços em background
    │   │   └── Registra PIDs
    │   │
    │   └── API_EXAMPLES.sh                # Exemplos de requisições
    │       ├── Predição single-step
    │       ├── Predição multi-step
    │       ├── Listar tickers
    │       └── Análise técnica
    │
    ├── 📂 Dados & Logs
    │   ├── data/
    │   │   └── NASDAQ100_Historical_Data.csv
    │   │       ├── 100+ tickers
    │   │       ├── Dados históricos
    │   │       ├── OHLCV (Open, High, Low, Close, Volume)
    │   │       └── 26MB de dados
    │   │
    │   ├── logs/                          # Logs da aplicação
    │   │   ├── .gitkeep
    │   │   └── (Criados em runtime: API.log, Dashboard.log, Interface.log)
    │   │
    │   └── monitoring_logs/               # Eventos e métricas
    │       ├── events.jsonl               # Eventos do sistema (estruturado)
    │       ├── metrics.jsonl              # Métricas de monitoramento
    │       └── .gitkeep
    │
    └── __pycache__/                       # Cache Python (não versionado)
```

### Descrição dos Componentes

| Componente | Descrição |
|-----------|-----------|
| **api.py** | API RESTful Flask na porta 5001: 9 endpoints, Swagger interativo, validação de entrada |
| **web_interface.py** | Dashboard web Flask na porta 8080: 4 cards de funcionalidades, interface em português |
| **lstm_model.py** | Modelo LSTM com 52K parâmetros: 2 camadas LSTM (128/64 unidades), sequência de 60 dias, acurácia ~85% |
| **lstm_inference.py** | Engine de inferência: carregamento de modelo, normalização Min-Max, predição single/multi-step |
| **monitoring_dashboard.py** | Dashboard métricas Flask porta 5000: latência, CPU/memória, histórico, drift detection |
| **model_monitoring.py** | Monitor automático: drift detection, degradação de performance, health checks, logs JSONL |
| **requirements.txt** | 15+ pacotes: TensorFlow 2.13+, Flask 2.3+, Pandas, NumPy, Scikit-learn, Flasgger |
| **swagger.yml** | OpenAPI 3.0: 9 endpoints documentados, 5 categorias, schemas e exemplos completos |
---

## 🐳 Execução com Podman

> **Nota:** A partir da v2.0, a aplicação é **completamente efêmera** e não requer volumes persistentes!  

### Pré-requisitos

- **Podman** ou **Docker** instalado
- Portas **8080**, **5001** e **5000** disponíveis
- Mínimo **2GB RAM** disponível

### 1. Build da Imagem

```bash
# Clone o repositório
git clone https://github.com/Machine-Learning-Engineering/techchallenge_04.git
cd techchallenge_04

# Build da imagem
podman build -t lstm-prediction:latest -f Containerfile .
```

**Build inclui:** Python 3.11-slim, TensorFlow 2.13+, Flask, modelo pré-treinado, dados NASDAQ-100. Tempo: 5-10 min.

### Executar Container

**Simples:**
```bash
podman run -d --name lstm-prediction -p 8080:8080 -p 5001:5001 -p 5000:5000 lstm-prediction:latest
```

**Verificar:**
```bash
# Logs
podman logs -f lstm-prediction

# Health check
curl http://localhost:5001/health

# Parar
podman stop lstm-prediction && podman rm lstm-prediction
```

---

## 🔧 Troubleshooting

### Erro: "Permission denied"
O sistema foi configurado para resolver automaticamente:
- ✅ Fallback automático para `/tmp` se houver problemas de permissão
- ✅ Console logging como fallback final
- ℹ️ Não é necessário usar volumes especiais ou flags de usuário

### Outros Erros Comuns

**"Model not loaded"**: Verificar se `lstm_model_AAPL.keras` existe e se há memória suficiente (mín. 2GB)

**"Port already in use"**: Usar portas diferentes com `-p 8081:8080 -p 5002:5001 -p 5001:5000`

**Predições lentas (>500ms)**: Limitar recursos com `--memory="2g" --cpus="2"` ou verificar carga do sistema

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

## 👥 Contribuidores

- **Tech Challenge FIAP** - Desenvolvimento inicial

---

## 📞 Suporte

Para dúvidas ou problemas:

1. Verifique a [documentação](#-sumário)
2. Consulte o [Troubleshooting](#-troubleshooting)
3. Veja a [estrutura do projeto](#-estrutura-do-projeto)

---

**Versão**: 1.0.0  
**Última atualização**: 26 de fevereiro de 2026  
**Status**: ✅ Em produção
