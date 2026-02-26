#!/bin/bash
# API_EXAMPLES.sh - Exemplos rápidos de cURL para a API LSTM

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
API_URL="http://localhost:5001"
TICKER="AAPL"
DAYS=60

echo "🚀 LSTM Stock Price API - Exemplos de cURL"
echo "================================================"
echo ""
echo "API URL: $API_URL"
echo "Ticker padrão: $TICKER"
echo "."
echo ""

# ============================================================================
# 1. HEALTH CHECK
# ============================================================================
echo "1️⃣  Health Check"
echo "   Verifica se a API está rodando"
echo ""
curl -s "$API_URL/health" | json_pp
echo ""
echo ""

# ============================================================================
# 2. API INFO
# ============================================================================
echo "2️⃣  API Info"
echo "   Informações da API e modelo"
echo ""
curl -s "$API_URL/api/v1/info" | json_pp | head -30
echo "   (... truncado ...)"
echo ""
echo ""

# ============================================================================
# 3. LIST TICKERS
# ============================================================================
echo "3️⃣  List Tickers"
echo "   Lista todos os 103 tickers disponíveis"
echo ""
curl -s "$API_URL/api/v1/tickers" | json_pp
echo ""
echo ""

# ============================================================================
# 4. GET TICKER INFO
# ============================================================================
echo "4️⃣  Get Ticker Info"
echo "   Informações detalhadas de um ticker"
echo ""
echo "   cURL: GET /api/v1/ticker/AAPL"
curl -s "$API_URL/api/v1/ticker/AAPL" | json_pp
echo ""
echo ""

# ============================================================================
# 5. SINGLE-STEP PREDICTION (Ticker)
# ============================================================================
echo "5️⃣  Single-Step Prediction (com Ticker)"
echo "   Prediz o preço do próximo dia usando dados históricos"
echo ""
echo "   cURL: POST /api/v1/predict"
echo "   Dados: {\"ticker\": \"AAPL\", \"days\": 60}"
echo ""
curl -s -X POST "$API_URL/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "days": 60}' | json_pp
echo ""
echo ""

# ============================================================================
# 6. SINGLE-STEP PREDICTION (Prices)
# ============================================================================
echo "6️⃣  Single-Step Prediction (com Array de Preços)"
echo "   Prediz usando um array de preços customizado"
echo ""
echo "   cURL: POST /api/v1/predict"
echo "   Dados: {\"prices\": [...]}"
echo ""
curl -s -X POST "$API_URL/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "prices": [263.5, 264.0, 265.2, 264.8, 266.1, 266.5, 267.0, 266.7, 268.2, 269.0,
               270.5, 269.8, 271.2, 270.9, 272.1, 273.0, 272.5, 273.8, 274.5, 275.2,
               274.9, 276.1, 277.0, 276.5, 278.2, 279.1, 279.5, 280.1, 279.8, 281.0,
               282.1, 281.5, 283.0, 284.2, 284.8, 285.5, 286.0, 285.7, 287.1, 288.0,
               287.5, 289.1, 290.0, 289.7, 291.2, 292.0, 291.8, 293.1, 294.0, 293.9,
               295.2, 296.0, 295.7, 297.1, 298.0, 297.8, 299.1, 300.0, 299.5, 301.0]
  }' | json_pp
echo ""
echo ""

# ============================================================================
# 7. MULTI-STEP PREDICTION (15 dias)
# ============================================================================
echo "7️⃣  Multi-Step Prediction (15 dias)"
echo "   Prediz preços para os próximos 15 dias"
echo ""
echo "   cURL: POST /api/v1/predict-multi"
echo "   Dados: {\"ticker\": \"AAPL\", \"forecast_days\": 15}"
echo ""
curl -s -X POST "$API_URL/api/v1/predict-multi" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "days": 60, "forecast_days": 15}' | json_pp
echo ""
echo ""

# ============================================================================
# 8. MONITORING REPORT
# ============================================================================
echo "8️⃣  Monitoring Report"
echo "   Métricas de performance da API"
echo ""
echo "   cURL: GET /api/v1/monitoring/report"
echo ""
curl -s "$API_URL/api/v1/monitoring/report" | json_pp
echo ""
echo ""

# ============================================================================
# 9. TESTE DE STRESSE (5 requisições rápidas)
# ============================================================================
echo "9️⃣  Stress Test (5 requisições)"
echo "   Testa a performance sob múltiplas requisições"
echo ""
echo "   Enviando 5 requisições para /api/v1/predict..."
echo ""

total_time=0
for i in {1..5}; do
    start=$(date +%s%N)
    response=$(curl -s -X POST "$API_URL/api/v1/predict" \
      -H "Content-Type: application/json" \
      -d '{"ticker": "AAPL", "days": 60}')
    end=$(date +%s%N)
    
    latency=$((($end - $start) / 1000000))
    total_time=$((total_time + latency))
    
    price=$(echo $response | grep -o '"predicted_price": [0-9.]*' | cut -d' ' -f2)
    echo "   Requisição $i: ${latency}ms - Preço predito: \$$price"
done

avg_time=$((total_time / 5))
echo ""
echo "   Latência média: ${avg_time}ms"
echo ""
echo ""

# ============================================================================
# 10. EXEMPLO: DIFERENTES TICKERS
# ============================================================================
echo "🔟 Predictions para Múltiplos Tickers"
echo "    Predições rápidas para os 5 maiores stocks"
echo ""

for ticker in AAPL MSFT GOOGL AMZN NVDA; do
    echo "   $ticker..."
    response=$(curl -s -X POST "$API_URL/api/v1/predict" \
      -H "Content-Type: application/json" \
      -d "{\"ticker\": \"$ticker\", \"days\": 60}")
    
    pred_price=$(echo $response | grep -o '"predicted_price": [0-9.]*' | cut -d' ' -f2)
    change=$(echo $response | grep -o '"change_pct": -\?[0-9.]*' | cut -d' ' -f2)
    
    printf "     Próximo dia: \$%-8.2f | Mudança: %+7.2f%%\n" "$pred_price" "$change"
done

echo ""
echo ""

# ============================================================================
# 11. INFORMAÇÕES PARA DESENVOLVIMENTO
# ============================================================================
echo "📝 Informações para Desenvolvimento"
echo "================================================"
echo ""
echo "Exemplos de JSON para requisições:"
echo ""
echo "Single prediction (ticker):"
echo '  {"ticker": "AAPL", "days": 60}'
echo ""
echo "Single prediction (prices):"
echo '  {"prices": [263.5, 264.0, 265.2, ...]} (min 10 preços)'
echo ""
echo "Multi-step prediction:"
echo '  {"ticker": "AAPL", "days": 60, "forecast_days": 15}'
echo ""
echo "URL para acessar no navegador:"
echo "  $API_URL/health"
echo "  $API_URL/api/v1/tickers"
echo "  $API_URL/api/v1/ticker/AAPL"
echo "  $API_URL/api/v1/monitoring/report"
echo ""
echo ""

echo "✅ Teste completo finalizado!"
echo ""
