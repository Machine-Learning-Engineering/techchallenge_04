#!/bin/bash
# ============================================================================
# RUN ALL - Inicia todos os serviços do projeto LSTM
# ============================================================================
# Uso: bash run_all.sh
# 
# Serviços iniciados:
#   1. API LSTM (porta 5001)
#   2. Dashboard de Monitoramento (porta 5000)
#   3. Interface Web Única (porta 8080)
#
# Para parar todos: Ctrl+C ou bash run_all.sh stop
# ============================================================================

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Diretórios
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR"
PIDS_FILE="$PROJECT_DIR/.service_pids"

# Função de log
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Função para verificar se porta está em uso
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Porta em uso
    else
        return 1  # Porta livre
    fi
}

# Função para matar processo em uma porta
kill_port() {
    local port=$1
    log_info "Verificando porta $port..."
    if check_port $port; then
        log_warning "Porta $port em uso. Tentando liberar..."
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 1
        if check_port $port; then
            log_error "Não foi possível liberar porta $port"
            return 1
        else
            log_success "Porta $port liberada"
        fi
    fi
    return 0
}

# Função para parar todos os serviços
stop_services() {
    log_info "Parando todos os serviços..."
    
    if [ -f "$PIDS_FILE" ]; then
        while read pid; do
            if ps -p $pid > /dev/null 2>&1; then
                log_info "Parando processo $pid..."
                kill -15 $pid 2>/dev/null || true
            fi
        done < "$PIDS_FILE"
        rm -f "$PIDS_FILE"
    fi
    
    # Garantir que as portas estão livres
    kill_port 5001  # API
    kill_port 5000  # Dashboard
    kill_port 8080  # Interface
    
    log_success "Todos os serviços parados"
}

# Função para verificar Python e dependências
check_dependencies() {
    log_info "Verificando dependências..."
    
    # Verificar Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 não encontrado. Instale Python 3.7+"
        exit 1
    fi
    
    # Verificar dependências Python
    python3 -c "import flask, flask_cors, requests, tensorflow, pandas, flasgger, yaml" 2>/dev/null
    if [ $? -ne 0 ]; then
        log_warning "Algumas dependências Python não encontradas"
        log_info "Instalando dependências..."
        pip install -q -r requirements.txt
    fi
    
    log_success "Dependências OK"
}

# Função para verificar arquivos necessários
check_files() {
    log_info "Verificando arquivos necessários..."
    
    local required_files=(
        "api.py"
        "monitoring_dashboard.py"
        "web_interface.py"
        "lstm_model_AAPL.keras"
        "data/NASDAQ100_Historical_Data.csv"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$PROJECT_DIR/$file" ]; then
            log_error "Arquivo não encontrado: $file"
            exit 1
        fi
    done
    
    log_success "Todos os arquivos encontrados"
}

# Função para iniciar um serviço em background
start_service() {
    local name=$1
    local port=$2
    local script=$3
    local log_file="$PROJECT_DIR/logs/${name}.log"
    
    # Criar diretório de logs se não existir
    mkdir -p "$PROJECT_DIR/logs"
    
    log_info "Iniciando $name (porta $port)..."
    
    # Iniciar serviço
    cd "$PROJECT_DIR"
    nohup python3 "$script" > "$log_file" 2>&1 &
    local pid=$!
    
    # Salvar PID
    echo $pid >> "$PIDS_FILE"
    
    # Aguardar serviço iniciar
    local max_wait=30
    local wait_count=0
    while ! check_port $port; do
        sleep 1
        wait_count=$((wait_count + 1))
        if [ $wait_count -ge $max_wait ]; then
            log_error "$name não iniciou em ${max_wait}s"
            log_error "Verifique o log: $log_file"
            return 1
        fi
    done
    
    log_success "$name iniciado (PID: $pid, porta: $port)"
    echo "  📄 Log: $log_file"
}

# Função para mostrar status
show_status() {
    echo ""
    echo "============================================================================"
    echo -e "${GREEN}✅ Todos os serviços estão rodando!${NC}"
    echo "============================================================================"
    echo ""
    echo "🌐 Acesse os serviços:"
    echo ""
    echo -e "  ${BLUE}Interface Web Única:${NC}"
    echo "    → http://localhost:8080"
    echo "    (Dashboard completo com todas as funcionalidades)"
    echo ""
    echo -e "  ${BLUE}API LSTM:${NC}"
    echo "    → http://localhost:5001"
    echo "    Endpoints: /health, /api/v1/predict, /api/v1/predict-multi"
    echo ""
    echo -e "  ${BLUE}Dashboard Monitoramento:${NC}"
    echo "    → http://localhost:5000"
    echo "    (Métricas de inferência e recursos)"
    echo ""
    echo "============================================================================"
    echo ""
    echo "📊 Quick Commands:"
    echo ""
    echo "  # Testar API"
    echo "  curl http://localhost:5001/health"
    echo ""
    echo "  # Fazer predição"
    echo "  python3 quick_client.py predict AAPL"
    echo ""
    echo "  # Ver logs"
    echo "  tail -f logs/*.log"
    echo ""
    echo "  # Parar serviços"
    echo "  bash run_all.sh stop"
    echo ""
    echo "============================================================================"
    echo ""
    echo -e "${YELLOW}Pressione Ctrl+C para parar todos os serviços${NC}"
    echo ""
}

# Função para aguardar sinal de interrupção
wait_for_interrupt() {
    # Aguardar Ctrl+C
    trap 'echo ""; log_info "Recebido sinal de interrupção..."; stop_services; exit 0' INT TERM
    
    # Loop infinito aguardando
    while true; do
        sleep 1
        
        # Verificar se serviços ainda estão rodando
        if [ -f "$PIDS_FILE" ]; then
            while read pid; do
                if ! ps -p $pid > /dev/null 2>&1; then
                    log_error "Serviço (PID $pid) parou inesperadamente"
                    stop_services
                    exit 1
                fi
            done < "$PIDS_FILE"
        fi
    done
}

# Função principal
main() {
    cd "$PROJECT_DIR"
    
    echo ""
    echo "============================================================================"
    echo "  🚀 LSTM Stock Price Prediction - RUN ALL"
    echo "============================================================================"
    echo ""
    
    # Verificar modo
    if [ "$1" == "stop" ]; then
        stop_services
        exit 0
    fi
    
    # Limpar PIDs antigos
    rm -f "$PIDS_FILE"
    
    # Verificações
    check_dependencies
    check_files
    
    # Liberar portas
    log_info "Liberando portas..."
    kill_port 5001
    kill_port 5000
    kill_port 8080
    
    echo ""
    log_info "Iniciando serviços..."
    echo ""
    
    # Iniciar serviços
    start_service "API" 5001 "api.py" || exit 1
    sleep 2
    
    start_service "Dashboard" 5000 "monitoring_dashboard.py" || exit 1
    sleep 2
    
    start_service "Interface" 8080 "web_interface.py" || exit 1
    sleep 2
    
    # Mostrar status
    show_status
    
    # Aguardar interrupção
    wait_for_interrupt
}

# Executar
main "$@"
