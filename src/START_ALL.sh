#!/bin/bash

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 LSTM Stock Price Prediction - INÍCIO RÁPIDO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✨ Este script iniciará automaticamente:"
echo ""
echo "   1️⃣  API LSTM (porta 5001)"
echo "   2️⃣  Dashboard de Monitoramento (porta 5000)"
echo "   3️⃣  Interface Web Completa (porta 8080) ⭐"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Iniciando em 3 segundos..."
sleep 1
echo "2..."
sleep 1
echo "1..."
sleep 1
echo ""
echo "🔥 INICIANDO TODOS OS SERVIÇOS..."
echo ""

# Mudar para o diretório do script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
bash run_all.sh
