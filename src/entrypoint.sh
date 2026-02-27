#!/bin/bash
set -e

# Criar diretórios efêmeros com permissões adequadas
mkdir -p /tmp/lstm-logs /tmp/lstm-monitoring-logs

# Criar symlinks dos diretórios de logs para tmpfs
# Isso garante que os logs sejam armazenados em memória (efêmero)
rm -rf /app/src/logs /app/src/monitoring_logs 2>/dev/null || true
ln -sf /tmp/lstm-logs /app/src/logs
ln -sf /tmp/lstm-monitoring-logs /app/src/monitoring_logs

# Iniciar os serviços
exec bash /app/src/START_ALL.sh
