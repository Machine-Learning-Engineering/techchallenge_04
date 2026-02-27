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
  - [Árvore Completa](#árvore-completa)
  - [Descrição dos Componentes](#descrição-dos-componentes)
- [Execução com Podman](#-execução-com-podman)
  - [Build da Imagem](#1-build-da-imagem)
  - [Executar Container](#2-executar-container)
  - [Verificar Status](#3-verificar-status)
- [APIs](#-apis)
  - [Endpoints Disponíveis](#endpoints-disponíveis)
  - [Exemplos de Uso](#exemplos-de-uso)
  - [Documentação Swagger](#documentação-swagger)
- [Monitoramento](#-monitoramento)
  - [Dashboard](#dashboard)
  - [Métricas Coletadas](#métricas-coletadas)
  - [Health Check](#health-check)
- [Logs](#-logs)
  - [Tipos de Logs](#tipos-de-logs)
  - [Localização](#localização)
  - [Acesso via API](#acesso-via-api)
- [Desenvolvimento Local](#-desenvolvimento-local)
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

#### 🔌 API & Interface (api.py, web_interface.py)
- **API RESTful**: Serviço principal em Flask na porta 5001
  - 9 endpoints para predição, análise e dados
  - Documentação Swagger interativa
  - Validação de entrada e tratamento de erros
  
- **Web Interface**: Dashboard web na porta 8080
  - Interface amigável em português
  - 4 cards de funcionalidades
  - Links para Swagger e documentação
  - Responsive design (desktop/mobile)

#### 🤖 Machine Learning (lstm_model.py, lstm_inference.py)
- **Modelo LSTM**: Rede neural treinada com 52,033 parâmetros
  - 2 camadas LSTM (128 e 64 unidades)
  - Sequência de entrada: 60 dias
  - Output: Previsão do próximo dia
  - Taxa de acurácia: ~85%
  
- **Engine de Inferência**: Orquestra predições
  - Carregamento automático do modelo
  - Normalização Min-Max dos dados
  - Cálculo de métricas (volatilidade, tendência, confiança)
  - Suporte a single-step e multi-step

#### 📊 Monitoramento (monitoring_dashboard.py, model_monitoring.py)
- **Dashboard de Métricas**: Visualização em tempo real
  - Latência de inferência
  - Uso de CPU/memória
  - Histórico de predições
  - Alertas e condições anormais
  
- **Monitor do Modelo**: Detecção automática
  - Drift detection (mudanças no dataset)
  - Degradação de performance
  - Health checks periódicos
  - Logs estruturados em JSONL

#### ⚙️ Configuração (requirements.txt, swagger.yml, scripts)
- **Dependências**: 15+ pacotes Python
- **OpenAPI**: Especificação completa de todos os endpoints
- **Scripts**: Automação de inicialização e orquestração

#### 📂 Dados & Logs
- **Dataset**: NASDAQ-100 histórico com 26MB
- **Logs**: Estruturados em JSON Lines para análise posterior
- **Diretórios**: Estrutura preparada para volumes em container

### Fluxo de Dados

```
Cliente (Browser/cURL)
    ↓
API RESTful (5001)
    ↓
├→ Validação de entrada
│
├→ LSTM Inference Engine
│   ├→ Carrega modelo
│   ├→ Normaliza dados históricos
│   ├→ Executa predição
│   └→ Calcula métricas
│
├→ Monitoring System
│   ├→ Registra latência
│   ├→ Coleta recursos do SO
│   ├→ Detecta anomalias
│   └→ Log em eventos.jsonl
│
└→ Response (JSON)
    ↓
Cliente (resultado + metadata)
```

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

**O build inclui:**
- ✅ Python 3.11-slim
- ✅ TensorFlow 2.13+
- ✅ Flask e dependências
- ✅ Modelo LSTM pré-treinado
- ✅ **Dados históricos NASDAQ-100 inclusos**
- ✅ Scripts de inicialização

**Tempo estimado:** 5-10 minutos (primeira vez)

### 2. Executar Container 

#### ✨ Forma Simples (Recomendado)

```bash
# Sem nenhum volume - tudo automaticamente efêmero
podman run -d \
  --name lstm-prediction \
  -p 8080:8080 \
  -p 5001:5001 \
  -p 5000:5000 \
  lstm-prediction:latest
```


### 3. Verificar Status

```bash
# Ver containers rodando
podman ps

# Ver logs em tempo real
podman logs -f lstm-prediction

# Verificar health check
podman inspect lstm-prediction | grep -A 5 Health

# Acessar shell do container
podman exec -it lstm-prediction bash
```

**Validação de Serviços:**

```bash
# Health check da API
curl http://localhost:5001/health

# Listar tickers disponíveis
curl http://localhost:5001/api/v1/tickers

# Verificar web interface
curl -I http://localhost:8080
```

**Parar e Remover:**

```bash
# Parar container
podman stop lstm-prediction

# Remover container
podman rm lstm-prediction
```

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
  - [Árvore Completa](#árvore-completa)
  - [Descrição dos Componentes](#descrição-dos-componentes)
- [Execução com Podman](#-execução-com-podman)
  - [Build da Imagem](#1-build-da-imagem)
  - [Executar Container](#2-executar-container)
  - [Verificar Status](#3-verificar-status)
- [APIs](#-apis)
  - [Endpoints Disponíveis](#endpoints-disponíveis)
  - [Exemplos de Uso](#exemplos-de-uso)
  - [Documentação Swagger](#documentação-swagger)
- [Monitoramento](#-monitoramento)
  - [Dashboard](#dashboard)
  - [Métricas Coletadas](#métricas-coletadas)
  - [Health Check](#health-check)
- [Logs](#-logs)
  - [Tipos de Logs](#tipos-de-logs)
  - [Localização](#localização)
  - [Acesso via API](#acesso-via-api)
- [Desenvolvimento Local](#-desenvolvimento-local)
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

#### 🔌 API & Interface (api.py, web_interface.py)
- **API RESTful**: Serviço principal em Flask na porta 5001
  - 9 endpoints para predição, análise e dados
  - Documentação Swagger interativa
  - Validação de entrada e tratamento de erros
  
- **Web Interface**: Dashboard web na porta 8080
  - Interface amigável em português
  - 4 cards de funcionalidades
  - Links para Swagger e documentação
  - Responsive design (desktop/mobile)

#### 🤖 Machine Learning (lstm_model.py, lstm_inference.py)
- **Modelo LSTM**: Rede neural treinada com 52,033 parâmetros
  - 2 camadas LSTM (128 e 64 unidades)
  - Sequência de entrada: 60 dias
  - Output: Previsão do próximo dia
  - Taxa de acurácia: ~85%
  
- **Engine de Inferência**: Orquestra predições
  - Carregamento automático do modelo
  - Normalização Min-Max dos dados
  - Cálculo de métricas (volatilidade, tendência, confiança)
  - Suporte a single-step e multi-step

#### 📊 Monitoramento (monitoring_dashboard.py, model_monitoring.py)
- **Dashboard de Métricas**: Visualização em tempo real
  - Latência de inferência
  - Uso de CPU/memória
  - Histórico de predições
  - Alertas e condições anormais
  
- **Monitor do Modelo**: Detecção automática
  - Drift detection (mudanças no dataset)
  - Degradação de performance
  - Health checks periódicos
  - Logs estruturados em JSONL

#### ⚙️ Configuração (requirements.txt, swagger.yml, scripts)
- **Dependências**: 15+ pacotes Python
- **OpenAPI**: Especificação completa de todos os endpoints
- **Scripts**: Automação de inicialização e orquestração

#### 📂 Dados & Logs
- **Dataset**: NASDAQ-100 histórico com 26MB
- **Logs**: Estruturados em JSON Lines para análise posterior
- **Diretórios**: Estrutura preparada para volumes em container

### Fluxo de Dados

```
Cliente (Browser/cURL)
    ↓
API RESTful (5001)
    ↓
├→ Validação de entrada
│
├→ LSTM Inference Engine
│   ├→ Carrega modelo
│   ├→ Normaliza dados históricos
│   ├→ Executa predição
│   └→ Calcula métricas
│
├→ Monitoring System
│   ├→ Registra latência
│   ├→ Coleta recursos do SO
│   ├→ Detecta anomalias
│   └→ Log em eventos.jsonl
│
└→ Response (JSON)
    ↓
Cliente (resultado + metadata)
```

---

## �🐳 Execução com Podman

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

**Parâmetros do Build:**
- `-t lstm-prediction:latest`: Nome e tag da imagem
- `-f Containerfile`: Arquivo de definição do container

**O build inclui:**
- Python 3.11-slim como base
- TensorFlow 2.13+
- Flask e dependências
- Modelo LSTM pré-treinado
- Dados históricos NASDAQ-100

**Tempo estimado:** 5-10 minutos (primeira vez)

### 2. Executar Container

```bash
# Executar em modo daemon
podman run \
  --replace \
  --name lstm-prediction \
  -p 8080:8080 \
  -p 5001:5001 \
  -p 5000:5000 \
  -v $(pwd)/src/logs:/app/src/logs:z \
  -v $(pwd)/src/monitoring_logs:/app/src/monitoring_logs:z \
  --restart unless-stopped \
  lstm-prediction:latest
```

**Parâmetros Explicados:**

| Parâmetro | Descrição |
|-----------|-----------|
| `-d` | Executa em background (daemon) |
| `--name` | Nome do container |
| `-p 8080:8080` | Porta da Web Interface |
| `-p 5001:5001` | Porta da API RESTful |
| `-p 5000:5000` | Porta do Monitoring Dashboard |
| `-v logs:...` | Volume para persistir logs da aplicação |
| `-v monitoring_logs:...` | Volume para persistir métricas |
| `--restart unless-stopped` | Restart automático em falhas |

### 3. Verificar Status

```bash
# Ver containers rodando
podman ps

# Ver logs em tempo real
podman logs -f lstm-prediction

# Verificar health check
podman inspect lstm-prediction | grep -A 5 Health

# Acessar shell do container
podman exec -it lstm-prediction bash
```

**Validação de Serviços:**

```bash
# Health check da API
curl http://localhost:5001/health

# Listar tickers disponíveis
curl http://localhost:5001/api/v1/tickers

# Verificar web interface
curl -I http://localhost:8080
```

**Parar e Remover:**

```bash
# Parar container
podman stop lstm-prediction

# Remover container
podman rm lstm-prediction

# Remover imagem
podman rmi lstm-prediction:latest
```

---

## 🔌 APIs

### Endpoints Disponíveis

#### **Sistema**

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/health` | Health check da aplicação |
| `GET` | `/api/v1/info` | Informações da API e endpoints |
| `GET` | `/api/v1/logs` | Logs do sistema (filtráveis) |

#### **Predições**

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `POST` | `/api/v1/predict` | Predição single-step (próximo dia) |
| `POST` | `/api/v1/predict-multi` | Predição multi-step (até 30 dias) |

#### **Dados**

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/api/v1/tickers` | Lista de tickers disponíveis |
| `GET` | `/api/v1/ticker/<ticker>` | Informações de um ticker |
| `GET` | `/api/v1/analytics/<ticker>` | Análise técnica completa |

#### **Monitoramento**

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/api/v1/monitoring/report` | Relatório de métricas |

### Exemplos de Uso

#### Predição Single-Step

```bash
curl -X POST http://localhost:5001/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "days": 60
  }'
```

**Resposta:**
```json
{
  "success": true,
  "prediction": {
    "ticker": "AAPL",
    "current_price": 264.35,
    "predicted_price": 270.95,
    "change": 6.60,
    "change_pct": 2.50,
    "confidence": "HIGH",
    "volatility": 0.0152,
    "trend": "🔺 UP"
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
```

#### Predição Multi-Step

```bash
curl -X POST http://localhost:5001/api/v1/predict-multi \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "days": 60,
    "forecast_days": 15
  }'
```

**Resposta:**
```json
{
  "success": true,
  "predictions": [
    {
      "day": 1,
      "predicted_price": 270.95,
      "change": 6.60,
      "change_pct": 2.50
    },
    {
      "day": 2,
      "predicted_price": 271.23,
      "change": 6.88,
      "change_pct": 2.60
    }
  ],
  "summary": {
    "total_days": 15,
    "initial_price": 264.35,
    "final_price": 273.23,
    "total_change": 8.88,
    "total_change_pct": 3.36,
    "trend": "upward"
  }
}
```

#### Listar Tickers

```bash
curl http://localhost:5001/api/v1/tickers
```

**Resposta:**
```json
{
  "success": true,
  "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "..."],
  "count": 100
}
```

#### Análise Técnica

```bash
curl "http://localhost:5001/api/v1/analytics/AAPL?period=90"
```

**Resposta:**
```json
{
  "success": true,
  "analytics": {
    "ticker": "AAPL",
    "period": 90,
    "dates": ["2025-11-01", "2025-11-02", "..."],
    "prices": [263.5, 264.0, 265.2],
    "volumes": [50000000, 55000000, 45000000],
    "volatility": [0.015, 0.018, 0.012],
    "returns": [0.0019, 0.0046, -0.0015]
  }
}
```

### Documentação Swagger

Acesse a documentação interativa em:

- **Swagger UI**: http://localhost:5001/swagger-ui/
- **Página de Docs**: http://localhost:5001/api-docs
- **OpenAPI JSON**: http://localhost:5001/openapi.json

**Specs por categoria:**
- Sistema: `/swagger/sistema.json`
- Predições: `/swagger/predicoes.json`
- Dados: `/swagger/dados.json`
- Análise: `/swagger/analise.json`
- Monitoramento: `/swagger/monitoramento.json`

**Testar via Swagger UI:**
1. Acesse http://localhost:5001/swagger-ui/
2. Selecione a categoria de API no dropdown
3. Escolha o endpoint desejado
4. Clique em "Try it out"
5. Preencha os parâmetros
6. Execute e veja a resposta

---

## 📊 Monitoramento

### Dashboard

Acesse o dashboard de monitoramento em: **http://localhost:5000**

O dashboard exibe:
- ✅ **Status**: Health do modelo e serviços
- 📈 **Métricas de Inferência**: Latência (média, min, max, p95)
- 💻 **Recursos do Sistema**: CPU, memória, disco
- 🎯 **Performance**: Total de predições, taxa de erro
- ⚠️ **Alertas**: Drift detection, anomalias

### Métricas Coletadas

#### Latência de Inferência
```json
{
  "metric": "inference_latency",
  "value": 45.3,
  "timestamp": "2026-02-26T16:10:00",
  "unit": "milliseconds"
}
```

#### Recursos do Sistema
```json
{
  "cpu_percent": 15.2,
  "memory_percent": 45.8,
  "memory_mb": 512.3,
  "disk_usage_percent": 35.1
}
```

#### Drift Detection
```json
{
  "drift_detected": false,
  "predictions_since_baseline": 150,
  "mean_drift": 0.02,
  "threshold": 0.15
}
```

### Health Check

O endpoint `/health` retorna o status da aplicação:

```bash
curl http://localhost:5001/health
```

**Resposta:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "uptime_seconds": 3600.5,
  "timestamp": "2026-02-26T16:10:00"
}
```

**Status possíveis:**
- `healthy`: Tudo operacional
- `unhealthy`: Modelo não carregado ou erro crítico

**Healthcheck do Container:**
- Intervalo: 30 segundos
- Timeout: 10 segundos
- Retries: 3
- Start period: 40 segundos

---

## 📝 Logs

### Tipos de Logs

#### 1. Logs de Eventos (`events.jsonl`)

Registra todos os eventos do sistema:

```json
{
  "timestamp": "2026-02-26T16:10:00.123Z",
  "event_type": "prediction",
  "message": "Predição realizada com sucesso",
  "details": {
    "ticker": "AAPL",
    "latency_ms": 45.3
  }
}
```

**Tipos de eventos:**
- `prediction`: Predições realizadas
- `inference`: Execuções do modelo
- `drift`: Detecções de drift
- `error`: Erros e exceções
- `warning`: Avisos do sistema
- `info`: Informações gerais

#### 2. Logs de Métricas (`metrics.jsonl`)

Registra métricas numéricas:

```json
{
  "timestamp": "2026-02-26T16:10:00.123Z",
  "metric": "inference_latency",
  "value": 45.3,
  "tags": {
    "model": "LSTM",
    "ticker": "AAPL"
  }
}
```

#### 3. Logs de Aplicação

Logs padrão dos serviços (stdout/stderr):
- `logs/API.log`: Logs da API
- `logs/Dashboard.log`: Logs do monitoring dashboard
- `logs/Interface.log`: Logs da web interface

### Localização

**No Container:**
- `/app/src/logs/` - Logs de aplicação
- `/app/src/monitoring_logs/` - Eventos e métricas

**No Host (volumes montados):**
- `./src/logs/` - Logs de aplicação
- `./src/monitoring_logs/` - Eventos e métricas

### Acesso via API

#### Listar Logs

```bash
curl "http://localhost:5001/api/v1/logs?limit=50&type=prediction"
```

**Parâmetros:**
- `limit`: Número de logs (default: 50, max: 1000)
- `type`: Filtrar por tipo (prediction, error, drift, etc)

**Resposta:**
```json
{
  "success": true,
  "logs": [
    {
      "timestamp": "2026-02-26T16:10:00",
      "type": "prediction",
      "message": "Predição realizada com sucesso"
    }
  ],
  "metrics": [
    {
      "timestamp": "2026-02-26T16:10:00",
      "metric": "inference_latency",
      "value": 45.3
    }
  ],
  "count": 50
}
```

#### Ver Logs do Container

```bash
# Logs em tempo real
podman logs -f lstm-prediction

# Últimas 100 linhas
podman logs --tail 100 lstm-prediction

# Logs desde timestamp
podman logs --since 2026-02-26T16:00:00 lstm-prediction
```

---

## 💻 Desenvolvimento Local

### Executar sem Container

```bash
# 1. Clonar repositório
git clone https://github.com/Machine-Learning-Engineering/techchallenge_04.git
cd techchallenge_04/src

# 2. Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Executar todos os serviços
./START_ALL.sh
```

**Serviços iniciados:**
- API: http://localhost:5001
- Web Interface: http://localhost:8080
- Monitoring: http://localhost:5000

### Tecnologias Utilizadas

| Tecnologia | Versão | Uso |
|------------|--------|-----|
| Python | 3.11 | Linguagem base |
| TensorFlow | 2.13+ | Framework de ML |
| Flask | 2.3+ | Framework web |
| Pandas | 1.5+ | Manipulação de dados |
| NumPy | 1.24+ | Computação numérica |
| Scikit-learn | 1.3+ | Pré-processamento |
| Flasgger | 0.9.7+ | Documentação Swagger |

---

## 🔧 Troubleshooting

### Container não inicia

```bash
# Ver logs de erro
podman logs lstm-prediction

# Verificar se portas estão em uso
lsof -i :8080 && lsof -i :5001 && lsof -i :5000

# Rebuild sem cache
podman build --no-cache -t lstm-prediction:latest -f Containerfile .
```

### Erro: "Model not loaded"

**Causa**: Modelo LSTM não foi carregado corretamente

**Solução**:
1. Verificar se `lstm_model_AAPL.keras` existe
2. Verificar logs: `podman logs lstm-prediction | grep "Modelo carregado"`
3. Verificar memória disponível (mínimo 2GB)

### Erro: "Port already in use"

**Causa**: Porta já está sendo usada

**Solução**:
```bash
# Encontrar processo usando a porta
lsof -i :8080

# Parar processo
kill -9 <PID>

# Ou usar portas diferentes
podman run -p 8081:8080 -p 5002:5001 -p 5001:5000 lstm-prediction:latest
```

### Predições lentas (> 500ms)

**Possíveis causas**:
- CPU sobrecarregada
- Memória insuficiente
- Muitas requisições simultâneas

**Solução**:
```bash
# Limitar recursos do container
podman run --memory="2g" --cpus="2" lstm-prediction:latest

# Verificar métricas
curl http://localhost:5001/api/v1/monitoring/report
```

### Health check falhando

```bash
# Verificar health manualmente
curl http://localhost:5001/health

# Ver tentativas de health check
podman inspect lstm-prediction | grep -A 10 Health

# Aumentar timeout
# Editar Containerfile e aumentar timeout em HEALTHCHECK
```

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
3. Veja os [logs](#-logs)
4. Acesse a [documentação Swagger](#documentação-swagger)

---

**Versão**: 1.0.0  
**Última atualização**: 26 de fevereiro de 2026  
**Status**: ✅ Em produção
