# ==============================================================================
# Containerfile - LSTM Stock Price Prediction
# ==============================================================================
# Build: podman build -t lstm-prediction .
# Run:   podman run -p 8080:8080 -p 5001:5001 -p 5000:5000 lstm-prediction
# ==============================================================================

FROM python:3.11-slim

LABEL maintainer="Tech Challenge FIAP"
LABEL description="LSTM Stock Price Prediction API with Web Interface and Monitoring"
LABEL version="1.0.0"

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_CPP_MIN_LOG_LEVEL=2

# Criar usuário não-root
RUN useradd -m -u 1000 -s /bin/bash appuser

# Diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    lsof \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primeiro (para cache de dependências)
COPY src/requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY src/ ./src/
COPY README.md .

# Criar diretórios necessários com permissões apropriadas
RUN mkdir -p /app/src/logs \
    /app/src/monitoring_logs \
    /app/src/data && \
    chmod 777 /app/src/logs \
    /app/src/monitoring_logs \
    /app/src/data

# Garantir que o script de inicialização seja executável
RUN chmod +x /app/src/START_ALL.sh /app/src/run_all.sh

# Mudar propriedade para appuser (appuser pode escrever em subdirs)
RUN chown -R appuser:appuser /app

# Trocar para usuário não-root
USER appuser

# Expor portas
EXPOSE 8080 5001 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Comando de inicialização
WORKDIR /app/src
CMD ["bash", "START_ALL.sh"]
