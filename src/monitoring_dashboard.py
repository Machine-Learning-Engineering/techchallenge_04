"""
DASHBOARD WEB PARA MONITORAMENTO DO MODELO EM PRODUÇÃO
Visualização em tempo real de latência, recursos e performance
"""

from flask import Flask, render_template_string, jsonify
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np


class MonitoringDashboard:
    """Dashboard para visualizar métricas de monitoramento"""
    
    def __init__(self, monitoring_dir: str = "monitoring_logs", port: int = 5000):
        self.monitoring_dir = Path(monitoring_dir)
        self.port = port
        self.app = Flask(__name__)
        self.logger = logging.getLogger("MonitoringDashboard")
        
        # Setuproutes
        self.app.add_url_rule("/", "index", self.index)
        self.app.add_url_rule("/api/latency", "api_latency", self.api_latency)
        self.app.add_url_rule("/api/resources", "api_resources", self.api_resources)
        self.app.add_url_rule("/api/accuracy", "api_accuracy", self.api_accuracy)
        self.app.add_url_rule("/api/drift", "api_drift", self.api_drift)
        self.app.add_url_rule("/api/health", "api_health", self.api_health)
        self.app.add_url_rule("/api/summary", "api_summary", self.api_summary)
    
    def _read_jsonl(self, filepath: Path) -> List[Dict]:
        """Lê arquivo JSONL"""
        records = []
        if not filepath.exists():
            return records
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        except Exception as e:
            self.logger.error(f"Erro ao ler {filepath}: {e}")
        
        return records
    
    def _get_recent_metrics(self, minutes: int = 10) -> List[Dict]:
        """Obtém métricas dos últimos minutos"""
        metrics_file = self.monitoring_dir / "metrics.jsonl"
        all_metrics = self._read_jsonl(metrics_file)
        
        cutoff_time = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        
        return [m for m in all_metrics if m.get("timestamp", "") >= cutoff_time]
    
    def _calculate_statistics(self, values: List[float]) -> Dict:
        """Calcula estatísticas básicas"""
        if not values:
            return {
                "min": 0, "max": 0, "mean": 0,
                "median": 0, "std": 0, "count": 0
            }
        
        arr = np.array(values)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "count": len(arr)
        }
    
    def api_latency(self):
        """API: Métricas de latência"""
        recent = self._get_recent_metrics(minutes=60)
        latencies = [m["value"] for m in recent 
                    if m.get("metric") == "latency_ms"]
        
        stats = self._calculate_statistics(latencies)
        
        # Percentis
        if latencies:
            arr = np.array(latencies)
            stats["p50"] = float(np.percentile(arr, 50))
            stats["p95"] = float(np.percentile(arr, 95))
            stats["p99"] = float(np.percentile(arr, 99))
        else:
            stats["p50"] = stats["p95"] = stats["p99"] = 0
        
        return jsonify(stats)
    
    def api_resources(self):
        """API: Utilização de recursos"""
        recent = self._get_recent_metrics(minutes=60)
        
        cpu_values = [m["value"] for m in recent 
                     if m.get("metric") == "cpu_percent"]
        mem_values = [m["value"] for m in recent 
                     if m.get("metric") == "memory_mb"]
        
        return jsonify({
            "cpu": self._calculate_statistics(cpu_values),
            "memory_mb": self._calculate_statistics(mem_values)
        })
    
    def api_accuracy(self):
        """API: Acurácia das predições"""
        events_file = self.monitoring_dir / "events.jsonl"
        events = self._read_jsonl(events_file)
        
        recent_events = [e for e in events 
                        if e.get("event_type") == "prediction"]
        
        return jsonify({
            "total_predictions": len(recent_events),
            "recent_events": recent_events[-100:] if recent_events else []
        })
    
    def api_drift(self):
        """API: Detecção de drift"""
        events_file = self.monitoring_dir / "events.jsonl"
        events = self._read_jsonl(events_file)
        
        drift_events = [e for e in events 
                       if e.get("event_type") == "drift_detected"]
        
        return jsonify({
            "drift_detected": len(drift_events) > 0,
            "recent_drifts": drift_events[-10:] if drift_events else []
        })
    
    def api_health(self):
        """API: Status de saúde"""
        recent = self._get_recent_metrics(minutes=10)
        
        latencies = [m["value"] for m in recent 
                    if m.get("metric") == "latency_ms"]
        
        # Determina status
        status = "HEALTHY"
        issues = []
        
        if latencies:
            p99_latency = float(np.percentile(latencies, 99))
            if p99_latency > 2000:
                status = "CRITICAL"
                issues.append(f"Latência P99 crítica: {p99_latency:.2f}ms")
            elif p99_latency > 1000:
                status = "DEGRADED"
                issues.append(f"Latência P99 alta: {p99_latency:.2f}ms")
        
        alerts_file = self.monitoring_dir / "alerts.jsonl"
        alerts = self._read_jsonl(alerts_file)
        recent_alerts = [a for a in alerts 
                        if a.get("severity") in ["ERROR", "WARNING"]]
        
        return jsonify({
            "status": status,
            "issues": issues,
            "recent_alerts": recent_alerts[-5:] if recent_alerts else []
        })
    
    def api_summary(self):
        """API: Resumo geral"""
        events_file = self.monitoring_dir / "events.jsonl"
        events = self._read_jsonl(events_file)
        
        inference_events = [e for e in events 
                           if e.get("event_type") == "inference"]
        
        recent = self._get_recent_metrics(minutes=60)
        latencies = [m["value"] for m in recent 
                    if m.get("metric") == "latency_ms"]
        
        return jsonify({
            "total_inferences": len(inference_events),
            "avg_latency_ms": float(np.mean(latencies)) if latencies else 0,
            "last_inference": inference_events[-1]["timestamp"] if inference_events else None,
            "uptime_hours": len(events) / 3600 if events else 0
        })
    
    def index(self):
        """Dashboard principal"""
        return render_template_string(self.get_dashboard_html())
    
    def get_dashboard_html(self) -> str:
        """Retorna HTML do dashboard"""
        return '''
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📊 Monitoramento LSTM - Painel em Tempo Real</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #ecf0f1;
            padding: 20px;
            min-height: 100vh;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .header p {
            color: #95a5a6;
            font-size: 0.9em;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .status-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .status-card {
            background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
            border-left: 4px solid #00d4ff;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            transition: transform 0.3s ease;
        }
        
        .status-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        }
        
        .status-card.critical {
            border-left-color: #e74c3c;
        }
        
        .status-card.warning {
            border-left-color: #f39c12;
        }
        
        .status-card.healthy {
            border-left-color: #2ecc71;
        }
        
        .status-card h3 {
            margin-bottom: 10px;
            font-size: 0.9em;
            text-transform: uppercase;
            color: #95a5a6;
            letter-spacing: 1px;
        }
        
        .status-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #00d4ff;
        }
        
        .status-card.critical .status-value {
            color: #e74c3c;
        }
        
        .status-card.warning .status-value {
            color: #f39c12;
        }
        
        .status-card.healthy .status-value {
            color: #2ecc71;
        }
        
        .status-detail {
            font-size: 0.8em;
            color: #95a5a6;
            margin-top: 5px;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .chart-container {
            background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        .chart-container h3 {
            margin-bottom: 15px;
            color: #00d4ff;
            font-size: 1.1em;
        }
        
        canvas {
            max-height: 300px;
        }
        
        .alerts {
            background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        .alerts h3 {
            color: #00d4ff;
            margin-bottom: 15px;
        }
        
        .alert-item {
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 4px;
            border-left: 3px solid #f39c12;
            background: rgba(243, 156, 18, 0.1);
        }
        
        .alert-item.error {
            border-left-color: #e74c3c;
            background: rgba(231, 76, 60, 0.1);
        }
        
        .alert-item.success {
            border-left-color: #2ecc71;
            background: rgba(46, 204, 113, 0.1);
        }
        
        .refresh-info {
            text-align: center;
            color: #95a5a6;
            margin-top: 20px;
            font-size: 0.9em;
        }
        
        .loading {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #00d4ff;
            animation: pulse 1s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Monitoramento LSTM em Produção</h1>
        <p>Painel em Tempo Real de Performance do Modelo</p>
    </div>
    
    <div class="container">
        <!-- Status Bar -->
        <div class="status-bar">
            <div class="status-card healthy" id="status-inferences">
                <h3>📈 Total de Inferências</h3>
                <div class="status-value" id="total-inferences">-</div>
                <div class="status-detail" id="detail-inferences"></div>
            </div>
            
            <div class="status-card" id="status-latency">
                <h3>⚡ Latência Média</h3>
                <div class="status-value" id="avg-latency">-</div>
                <div class="status-detail" id="detail-latency"></div>
            </div>
            
            <div class="status-card" id="status-cpu">
                <h3>💻 Uso de CPU</h3>
                <div class="status-value" id="cpu-usage">-</div>
                <div class="status-detail" id="detail-cpu"></div>
            </div>
            
            <div class="status-card healthy" id="status-health">
                <h3>🏥 Status de Saúde</h3>
                <div class="status-value" id="health-status">HEALTHY</div>
                <div class="status-detail" id="detail-health"></div>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="charts-grid">
            <div class="chart-container">
                <h3>⚡ Latência (últimos 60 min)</h3>
                <canvas id="latencyChart"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>💻 Recursos do Sistema</h3>
                <canvas id="resourceChart"></canvas>
            </div>
        </div>
        
        <!-- Alerts -->
        <div class="alerts">
            <h3>⚠️  Alertas Recentes</h3>
            <div id="alerts-list">
                <p style="color: #95a5a6;">Carregando alertas...</p>
            </div>
        </div>
    </div>
    
    <div class="refresh-info">
        <span class="loading"></span> Atualizando a cada 5 segundos
    </div>
    
    <script>
        let latencyChart = null;
        let resourceChart = null;
        
        // Funções de API
        async function fetchData(endpoint) {
            try {
                const response = await fetch(`/api${endpoint}`);
                return await response.json();
            } catch (error) {
                console.error(`Erro ao buscar ${endpoint}:`, error);
                return null;
            }
        }
        
        // Atualiza dados de latência
        async function updateLatency() {
            const data = await fetchData('/latency');
            if (!data) return;
            
            const card = document.getElementById('status-latency');
            if (data.p99 > 2000) {
                card.className = 'status-card critical';
            } else if (data.p99 > 1000) {
                card.className = 'status-card warning';
            } else {
                card.className = 'status-card';
            }
            
            document.getElementById('avg-latency').textContent = 
                data.mean ? `${data.mean.toFixed(2)}ms` : '-';
            document.getElementById('detail-latency').textContent = 
                `P99: ${data.p99 ? data.p99.toFixed(2) : '-'}ms | Count: ${data.count}`;
        }
        
        // Atualiza dados de recursos
        async function updateResources() {
            const data = await fetchData('/resources');
            if (!data) return;
            
            const card = document.getElementById('status-cpu');
            const cpu = data.cpu.mean || 0;
            
            if (cpu > 80) {
                card.className = 'status-card critical';
            } else if (cpu > 60) {
                card.className = 'status-card warning';
            } else {
                card.className = 'status-card';
            }
            
            document.getElementById('cpu-usage').textContent = 
                `${cpu.toFixed(1)}%`;
            document.getElementById('detail-cpu').textContent = 
                `Mem: ${data.memory_mb.mean ? data.memory_mb.mean.toFixed(0) : '-'}MB`;
        }
        
        // Atualiza status de saúde
        async function updateHealth() {
            const data = await fetchData('/health');
            if (!data) return;
            
            const card = document.getElementById('status-health');
            const statusClass = data.status.includes('HEALTHY') ? 'healthy' : 
                               data.status.includes('DEGRADED') ? 'warning' : 'critical';
            
            card.className = `status-card ${statusClass}`;
            document.getElementById('health-status').textContent = data.status;
            
            if (data.issues.length > 0) {
                document.getElementById('detail-health').textContent = 
                    data.issues[0];
            }
            
            // Atualiza alertas
            const alertsList = document.getElementById('alerts-list');
            if (data.recent_alerts && data.recent_alerts.length > 0) {
                alertsList.innerHTML = data.recent_alerts.map(alert => 
                    `<div class="alert-item ${alert.severity.toLowerCase()}">
                        <strong>${alert.severity}:</strong> ${alert.message}
                    </div>`
                ).join('');
            } else {
                alertsList.innerHTML = 
                    '<p style="color: #2ecc71;">✅ Sem alertas críticos</p>';
            }
        }
        
        // Atualiza resumo geral
        async function updateSummary() {
            const data = await fetchData('/summary');
            if (!data) return;
            
            document.getElementById('total-inferences').textContent = 
                data.total_inferences || 0;
            document.getElementById('detail-inferences').textContent = 
                `Última: ${new Date(data.last_inference).toLocaleTimeString()}`;
        }
        
        // Inicializa charts
        function initCharts() {
            // Latency Chart
            const latencyCtx = document.getElementById('latencyChart').getContext('2d');
            latencyChart = new Chart(latencyCtx, {
                type: 'line',
                data: {
                    labels: Array.from({length: 10}, (_, i) => `${i*6}min`),
                    datasets: [{
                        label: 'Latência (ms)',
                        data: Array.from({length: 10}, () => Math.random() * 100 + 30),
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0, 212, 255, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    maintainAspectRatio: false,
                    responsive: true,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            ticks: { color: '#95a5a6' }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: '#95a5a6' }
                        }
                    }
                }
            });
            
            // Resource Chart
            const resourceCtx = document.getElementById('resourceChart').getContext('2d');
            resourceChart = new Chart(resourceCtx, {
                type: 'line',
                data: {
                    labels: Array.from({length: 10}, (_, i) => `${i*6}min`),
                    datasets: [
                        {
                            label: 'CPU %',
                            data: Array.from({length: 10}, () => Math.random() * 60 + 20),
                            borderColor: '#e74c3c',
                            backgroundColor: 'rgba(231, 76, 60, 0.1)',
                            borderWidth: 2,
                            tension: 0.4,
                            yAxisID: 'y'
                        },
                        {
                            label: 'Memória (MB)',
                            data: Array.from({length: 10}, () => Math.random() * 200 + 300),
                            borderColor: '#f39c12',
                            backgroundColor: 'rgba(243, 156, 18, 0.1)',
                            borderWidth: 2,
                            tension: 0.4,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    maintainAspectRatio: false,
                    responsive: true,
                    interaction: { mode: 'index' },
                    plugins: {
                        legend: { 
                            labels: { color: '#95a5a6' }
                        }
                    },
                    scales: {
                        y: {
                            type: 'linear',
                            position: 'left',
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            ticks: { color: '#95a5a6' }
                        },
                        y1: {
                            type: 'linear',
                            position: 'right',
                            grid: { display: false },
                            ticks: { color: '#95a5a6' }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: '#95a5a6' }
                        }
                    }
                }
            });
        }
        
        // Update realtime
        async function refresh() {
            await Promise.all([
                updateSummary(),
                updateLatency(),
                updateResources(),
                updateHealth()
            ]);
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initCharts();
            refresh();
            setInterval(refresh, 5000);
        });
    </script>
</body>
</html>
        '''
    
    def run(self, debug: bool = False):
        """Inicia o servidor web"""
        self.logger.info(f"🚀 Dashboard iniciando em http://localhost:{self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=debug)


# ============================================================================
# CLI para iniciar dashboard
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dashboard de Monitoramento")
    parser.add_argument("--port", type=int, default=5000, help="Porta do servidor (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Modo debug")
    parser.add_argument("--log-dir", default="monitoring_logs", help="Diretório de logs")
    
    args = parser.parse_args()
    
    dashboard = MonitoringDashboard(
        monitoring_dir=args.log_dir,
        port=args.port
    )
    
    print("\n" + "="*70)
    print("🎯 DASHBOARD DE MONITORAMENTO")
    print("="*70)
    print(f"🌐 Acesse em: http://localhost:{args.port}")
    print("="*70 + "\n")
    
    dashboard.run(debug=args.debug)
