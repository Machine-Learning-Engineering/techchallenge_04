#!/bin/bash
set -e

# Criar diretórios efêmeros com permissões adequadas no /tmp
# /tmp é writable por todos os utilizadores normalmente
mkdir -p /tmp/lstm-logs /tmp/lstm-monitoring-logs

# Garantir permissões corretas para o usuário appuser (1000)
chmod 755 /tmp/lstm-logs /tmp/lstm-monitoring-logs 2>/dev/null || true

# Criar symlinks dos diretórios de logs para tmpfs
# Isso garante que os logs sejam armazenados em memória (efêmero)
rm -rf /app/src/logs /app/src/monitoring_logs 2>/dev/null || true
ln -sf /tmp/lstm-logs /app/src/logs
ln -sf /tmp/lstm-monitoring-logs /app/src/monitoring_logs

# Garantir que os diretórios locais também podem ser criados
mkdir -p /app/src/logs /app/src/monitoring_logs 2>/dev/null || true
chmod 755 /app/src/logs /app/src/monitoring_logs 2>/dev/null || true

echo "✅ Diretórios de log inicializados com sucesso"
echo "   - Logs: /app/src/logs (symlink para /tmp/lstm-logs)"
echo "   - Monitoring: /app/src/monitoring_logs (symlink para /tmp/lstm-monitoring-logs)"

# Iniciar os serviços
exec bash /app/src/START_ALL.sh
