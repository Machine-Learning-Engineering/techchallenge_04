"""
SISTEMA DE MONITORAMENTO PARA MODELO LSTM EM PRODUÇÃO
Rastreia performance, recursos, latência e detecção de drift do modelo
"""

import json
import logging
import time
import psutil
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from collections import deque


# ============================================================================
# Configurações de Logging
# ============================================================================

class MonitoringLogger:
    """Logger estruturado para monitoramento em produção"""
    
    def __init__(self, log_dir: str = "monitoring_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Logger para eventos estruturados (JSON)
        self.events_log = self.log_dir / "events.jsonl"
        
        # Logger para métricas de performance
        self.metrics_log = self.log_dir / "metrics.jsonl"
        
        # Logger para alertas
        self.alerts_log = self.log_dir / "alerts.jsonl"
        
        # Setup logging padrão
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "app.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("ModelMonitoring")
    
    def log_event(self, event_type: str, data: Dict):
        """Registra evento estruturado em JSON"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **data
        }
        with open(self.events_log, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def log_metric(self, metric_name: str, value: float, tags: Optional[Dict] = None):
        """Registra métrica de performance"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "metric": metric_name,
            "value": float(value),
            "tags": tags or {}
        }
        with open(self.metrics_log, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def log_alert(self, severity: str, message: str, details: Optional[Dict] = None):
        """Registra alerta"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "message": message,
            "details": details or {}
        }
        with open(self.alerts_log, "a") as f:
            f.write(json.dumps(record) + "\n")
        self.logger.warning(f"[{severity}] {message}")


# ============================================================================
# Métricas de Recursos do Sistema
# ============================================================================

@dataclass
class ResourceMetrics:
    """Métricas de utilização de recursos"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    timestamp: str


class SystemResourceMonitor:
    """Monitora utilização de CPU e memória"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.logger = logging.getLogger("SystemResourceMonitor")
    
    def get_metrics(self) -> ResourceMetrics:
        """Obtém métricas atuais de recursos"""
        cpu_percent = self.process.cpu_percent(interval=0.1)
        mem_info = self.process.memory_info()
        mem_percent = self.process.memory_percent()
        
        # Memória total do sistema
        vm = psutil.virtual_memory()
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=mem_percent,
            memory_used_mb=mem_info.rss / 1024 / 1024,
            memory_total_mb=vm.total / 1024 / 1024,
            timestamp=datetime.now().isoformat()
        )


# ============================================================================
# Métricas de Latência
# ============================================================================

@dataclass
class LatencyMetrics:
    """Métricas de latência de inferência"""
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    total_requests: int


class LatencyTracker:
    """Rastreia latência de inferências"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.logger = logging.getLogger("LatencyTracker")
    
    def record(self, latency_ms: float):
        """Registra uma latência em ms"""
        self.latencies.append(latency_ms)
    
    def get_metrics(self) -> LatencyMetrics:
        """Obtém estatísticas de latência"""
        if not self.latencies:
            return LatencyMetrics(0, 0, 0, 0, 0, 0, 0)
        
        arr = np.array(list(self.latencies))
        return LatencyMetrics(
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            mean_ms=float(np.mean(arr)),
            median_ms=float(np.median(arr)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            total_requests=len(self.latencies)
        )


# ============================================================================
# Detecção de Data Drift
# ============================================================================

class DataDriftDetector:
    """Detecta mudanças nas características dos dados de entrada"""
    
    def __init__(self, baseline_window: int = 50, threshold: float = 0.15):
        """
        Args:
            baseline_window: número de amostras para estabelecer baseline
            threshold: limiar para detectar drift (0-1)
        """
        self.baseline_window = baseline_window
        self.threshold = threshold
        self.baseline_stats = None
        self.current_batch = deque(maxlen=baseline_window)
        self.logger = logging.getLogger("DataDriftDetector")
    
    def record(self, prices: np.ndarray):
        """Registra preços para análise"""
        self.current_batch.append(prices)
    
    def set_baseline(self):
        """Estabelece baseline com dados atuais"""
        if len(self.current_batch) >= self.baseline_window:
            batch_array = np.array(list(self.current_batch))
            self.baseline_stats = {
                "mean": float(np.mean(batch_array)),
                "std": float(np.std(batch_array)),
                "min": float(np.min(batch_array)),
                "max": float(np.max(batch_array)),
                "timestamp": datetime.now().isoformat()
            }
            self.logger.info(f"Baseline estabelecido: {self.baseline_stats}")
    
    def detect_drift(self) -> Tuple[bool, Dict]:
        """
        Detecta drift comparando com baseline
        
        Returns:
            (drift_detected, drift_details)
        """
        if self.baseline_stats is None or len(self.current_batch) == 0:
            return False, {}
        
        current_array = np.array(list(self.current_batch))
        current_mean = float(np.mean(current_array))
        current_std = float(np.std(current_array))
        
        baseline_mean = self.baseline_stats["mean"]
        baseline_std = self.baseline_stats["std"] or 1
        
        # Calcula mudança percentual
        mean_change = abs(current_mean - baseline_mean) / abs(baseline_mean + 1e-10)
        std_change = abs(current_std - baseline_std) / abs(baseline_std + 1e-10)
        
        drift_detected = mean_change > self.threshold or std_change > self.threshold
        
        details = {
            "drift_detected": drift_detected,
            "mean_change_pct": float(mean_change * 100),
            "std_change_pct": float(std_change * 100),
            "threshold_pct": float(self.threshold * 100),
            "current_mean": current_mean,
            "baseline_mean": baseline_mean,
            "current_std": current_std,
            "baseline_std": baseline_std,
            "timestamp": datetime.now().isoformat()
        }
        
        return drift_detected, details


# ============================================================================
# Predição vs Realidade (Modelo Drift)
# ============================================================================

class PredictionAccuracyTracker:
    """Rastreia acurácia das predições vs valores reais"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.logger = logging.getLogger("PredictionAccuracyTracker")
    
    def record(self, predicted_price: float, actual_price: Optional[float] = None):
        """Registra predição e valor real (se disponível)"""
        self.predictions.append(predicted_price)
        if actual_price is not None:
            self.actuals.append(actual_price)
    
    def get_metrics(self) -> Dict:
        """Calcula métricas de acurácia"""
        if len(self.predictions) == 0:
            return {
                "mae": None,
                "mape": None,
                "rmse": None,
                "completed_predictions": 0
            }
        
        if len(self.actuals) < len(self.predictions):
            # Nem todas as predições têm valores reais ainda
            predictions_array = np.array(list(self.predictions)[:len(self.actuals)])
            actuals_array = np.array(list(self.actuals))
        else:
            predictions_array = np.array(list(self.predictions))
            actuals_array = np.array(list(self.actuals))
        
        if len(actuals_array) == 0:
            return {
                "mae": None,
                "mape": None,
                "rmse": None,
                "completed_predictions": 0
            }
        
        errors = predictions_array - actuals_array
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        
        # MAPE com proteção contra divisão por zero
        mape_values = np.abs(errors) / (np.abs(actuals_array) + 1e-10)
        mape = float(np.mean(mape_values) * 100)
        
        return {
            "mae": mae,
            "mape": mape,
            "rmse": rmse,
            "completed_predictions": len(actuals_array)
        }


# ============================================================================
# Agregador de Monitoramento
# ============================================================================

class ProductionModelMonitor:
    """Agrupa todos os monitores para rastreamento completo em produção"""
    
    def __init__(self, log_dir: str = "monitoring_logs"):
        self.logger = MonitoringLogger(log_dir)
        self.logger.logger.info("Iniciando Sistema de Monitoramento de Produção")
        
        self.resource_monitor = SystemResourceMonitor()
        self.latency_tracker = LatencyTracker(window_size=1000)
        self.drift_detector = DataDriftDetector(baseline_window=50)
        self.accuracy_tracker = PredictionAccuracyTracker(window_size=100)
        
        self.inference_count = 0
        self.start_time = datetime.now()
    
    def record_inference(self, latency_ms: float, resources_before=None):
        """Registra uma inferência realizada"""
        self.inference_count += 1
        self.latency_tracker.record(latency_ms)
        
        # Recursos atuais
        resources = self.resource_monitor.get_metrics()
        
        # Log da inferência
        self.logger.log_event("inference", {
            "inference_count": self.inference_count,
            "latency_ms": latency_ms,
            "cpu_percent": resources.cpu_percent,
            "memory_mb": resources.memory_used_mb
        })
        
        # Log de métrica individual
        self.logger.log_metric("latency_ms", latency_ms)
        self.logger.log_metric("cpu_percent", resources.cpu_percent)
        self.logger.log_metric("memory_mb", resources.memory_used_mb)
    
    def record_prediction(self, predicted_price: float, actual_price: Optional[float] = None):
        """Registra predição para rastreamento de acurácia"""
        self.accuracy_tracker.record(predicted_price, actual_price)
    
    def record_input_data(self, prices: np.ndarray):
        """Registra dados de entrada para análise de drift"""
        self.drift_detector.record(prices)
    
    def generate_report(self) -> Dict:
        """Gera relatório completo de monitoramento"""
        latency_metrics = self.latency_tracker.get_metrics()
        resource_metrics = self.resource_monitor.get_metrics()
        accuracy_metrics = self.accuracy_tracker.get_metrics()
        drift_detected, drift_details = self.drift_detector.detect_drift()
        
        uptime = datetime.now() - self.start_time
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime.total_seconds(),
            "inference_count": self.inference_count,
            
            "latency": asdict(latency_metrics),
            "resources": asdict(resource_metrics),
            "accuracy": accuracy_metrics,
            
            "data_drift": {
                "drift_detected": drift_detected,
                **drift_details
            },
            
            "health_status": self._determine_health_status(
                latency_metrics, accuracy_metrics, drift_detected
            )
        }
        
        return report
    
    def _determine_health_status(self, latency: LatencyMetrics, accuracy: Dict, 
                                  drift: bool) -> Dict:
        """Determina status de saúde do modelo em produção"""
        status = "HEALTHY"
        issues = []
        
        # Verifica latência
        if latency.p95_ms > 1000:
            status = "DEGRADED"
            issues.append(f"Latência P95 alta: {latency.p95_ms:.2f}ms")
        
        if latency.p99_ms > 2000:
            status = "CRITICAL"
            issues.append(f"Latência P99 crítica: {latency.p99_ms:.2f}ms")
        
        # Verifica acurácia
        if accuracy["mape"] and accuracy["mape"] > 10:
            status = "DEGRADED"
            issues.append(f"MAPE acurácia alta: {accuracy['mape']:.2f}%")
        
        # Verifica drift
        if drift:
            status = "DEGRADED"
            issues.append("Data drift detectado")
        
        return {
            "status": status,
            "issues": issues
        }
    
    def export_report(self, filepath: str):
        """Exporta relatório para arquivo JSON"""
        report = self.generate_report()
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        self.logger.logger.info(f"Relatório exportado: {filepath}")
    
    def print_summary(self):
        """Imprime resumo de monitoramento"""
        report = self.generate_report()
        
        print("\n" + "="*70)
        print("📊 RESUMO DE MONITORAMENTO - MODELO EM PRODUÇÃO")
        print("="*70)
        
        print(f"⏱️  Uptime: {report['uptime_seconds']:.2f}s")
        print(f"📈 Total de Inferências: {report['inference_count']}")
        
        latency = report["latency"]
        print(f"\n⚡ LATÊNCIA DE INFERÊNCIA:")
        print(f"   Min: {latency['min_ms']:.3f}ms")
        print(f"   Média: {latency['mean_ms']:.3f}ms")
        print(f"   P95: {latency['p95_ms']:.3f}ms")
        print(f"   P99: {latency['p99_ms']:.3f}ms")
        print(f"   Max: {latency['max_ms']:.3f}ms")
        
        resources = report["resources"]
        print(f"\n💻 UTILIZAÇÃO DE RECURSOS:")
        print(f"   CPU: {resources['cpu_percent']:.1f}%")
        print(f"   Memória: {resources['memory_used_mb']:.1f}MB / {resources['memory_total_mb']:.1f}MB ({resources['memory_percent']:.1f}%)")
        
        accuracy = report["accuracy"]
        if accuracy["completed_predictions"] > 0:
            print(f"\n🎯 ACURÁCIA DAS PREDIÇÕES:")
            print(f"   MAE: ${accuracy['mae']:.2f}")
            print(f"   MAPE: {accuracy['mape']:.2f}%")
            print(f"   RMSE: ${accuracy['rmse']:.2f}")
            print(f"   Predições Completas: {accuracy['completed_predictions']}")
        
        drift = report["data_drift"]
        print(f"\n🔄 DETECÇÃO DE DRIFT:")
        print(f"   Drift Detectado: {'SIM ⚠️' if drift['drift_detected'] else 'NÃO ✅'}")
        if "mean_change_pct" in drift:
            print(f"   Mudança Média: {drift['mean_change_pct']:.2f}%")
            print(f"   Mudança Desvio: {drift['std_change_pct']:.2f}%")
        
        health = report["health_status"]
        status_emoji = "✅" if health["status"] == "HEALTHY" else ("⚠️" if health["status"] == "DEGRADED" else "🔴")
        print(f"\n{status_emoji} STATUS DE SAÚDE: {health['status']}")
        if health["issues"]:
            for issue in health["issues"]:
                print(f"   ⚠️  {issue}")
        
        print("="*70 + "\n")


# ============================================================================
# Exemplo de Uso
# ============================================================================

if __name__ == "__main__":
    print("📊 SISTEMA DE MONITORAMENTO LSTM - MODO DEMO")
    print("="*70)
    
    # Inicializa monitor
    monitor = ProductionModelMonitor()
    
    print("✅ Monitor inicializado")
    print("📝 Gerando dados simulados de inferências...\n")
    
    # Simula inferências
    np.random.seed(42)
    for i in range(100):
        # Simula latência
        latency = np.random.normal(50, 15)
        latency = max(10, min(500, latency))
        
        # Simula preço predito e real
        actual_price = 264 + np.random.normal(0, 3)
        predicted_price = actual_price + np.random.normal(0, 5)
        
        # Simula dados de entrada
        prices = np.random.normal(264, 5, 60)
        
        # Registra
        monitor.record_inference(latency)
        monitor.record_prediction(predicted_price, actual_price if i > 20 else None)
        monitor.record_input_data(prices)
        
        # A cada 50 iterações estabelece baseline
        if i == 50:
            monitor.drift_detector.set_baseline()
    
    # Imprime resumo
    monitor.print_summary()
    
    # Exporta relatório
    report_path = "monitoring_report.json"
    monitor.export_report(report_path)
    print(f"📄 Relatório salvo em: {report_path}")
