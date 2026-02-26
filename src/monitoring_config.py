"""
CONFIGURAÇÃO DE MONITORAMENTO PARA PRODUÇÃO
Define thresholds, alertas e políticas de monitoramento
"""

from dataclasses import dataclass
from typing import Dict
from enum import Enum


class AlertSeverity(Enum):
    """Níveis de severidade de alerta"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SystemHealth(Enum):
    """Estados de saúde do sistema"""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


@dataclass
class LatencyThresholds:
    """Thresholds para métricas de latência"""
    # Latência máxima aceitável (ms)
    warning_p50: float = 100  # P50 acima disso = WARNING
    critical_p50: float = 500  # P50 acima disso = CRITICAL
    
    warning_p95: float = 500   # P95 acima disso = WARNING
    critical_p95: float = 2000  # P95 acima disso = CRITICAL
    
    warning_p99: float = 1000  # P99 acima disso = WARNING
    critical_p99: float = 5000  # P99 acima disso = CRITICAL
    
    max_permitted_spike: float = 3000  # Spike máximo em relação à média


@dataclass
class ResourceThresholds:
    """Thresholds para utilização de recursos"""
    # CPU (%)
    warning_cpu: float = 60.0
    critical_cpu: float = 85.0
    
    # Memória (%)
    warning_memory: float = 70.0
    critical_memory: float = 90.0
    
    # Memória absoluta (MB)
    critical_memory_mb: float = 2000.0  # Máximo de memória permitida


@dataclass
class AccuracyThresholds:
    """Thresholds para acurácia de predições"""
    # MAPE aceitável (%)
    warning_mape: float = 7.0
    critical_mape: float = 15.0
    
    # MAE aceitável ($)
    warning_mae: float = 15.0
    critical_mae: float = 50.0
    
    # R² mínimo aceitável
    minimum_r2: float = 0.85


@dataclass
class DriftThresholds:
    """Thresholds para detecção de drift"""
    # Mudança percentual permitida em estatísticas dos dados
    warning_threshold: float = 0.10  # 10%
    critical_threshold: float = 0.20  # 20%
    
    # Baseline samples para estabelecer referência
    baseline_window: int = 50


@dataclass
class MonitoringConfig:
    """Configuração completa de monitoramento"""
    
    # Diretório para armazenar logs de monitoramento
    log_directory: str = "monitoring_logs"
    
    # Frequência de logging (interpretado pelo aplicativo)
    log_frequency_seconds: int = 60
    
    # Thresholds
    latency_thresholds: LatencyThresholds = None
    resource_thresholds: ResourceThresholds = None
    accuracy_thresholds: AccuracyThresholds = None
    drift_thresholds: DriftThresholds = None
    
    # Comportamento de alertas
    enable_alerts: bool = True
    alert_email: str = ""  # Email para alertas críticos
    alert_webhook: str = ""  # URL para webhook de alertas
    
    # Retenção de dados
    log_retention_days: int = 30
    
    # Comportamento em degradação
    auto_fallback_on_degradation: bool = True
    fallback_model_path: str = ""
    
    def __post_init__(self):
        """Inicializa valores padrão para objetos dataclass aninhados"""
        if self.latency_thresholds is None:
            self.latency_thresholds = LatencyThresholds()
        if self.resource_thresholds is None:
            self.resource_thresholds = ResourceThresholds()
        if self.accuracy_thresholds is None:
            self.accuracy_thresholds = AccuracyThresholds()
        if self.drift_thresholds is None:
            self.drift_thresholds = DriftThresholds()
    
    def to_dict(self) -> Dict:
        """Converte configuração para dicionário"""
        return {
            "log_directory": self.log_directory,
            "log_frequency_seconds": self.log_frequency_seconds,
            
            "latency_thresholds": {
                "warning_p50": self.latency_thresholds.warning_p50,
                "critical_p50": self.latency_thresholds.critical_p50,
                "warning_p95": self.latency_thresholds.warning_p95,
                "critical_p95": self.latency_thresholds.critical_p95,
                "warning_p99": self.latency_thresholds.warning_p99,
                "critical_p99": self.latency_thresholds.critical_p99,
            },
            
            "resource_thresholds": {
                "warning_cpu": self.resource_thresholds.warning_cpu,
                "critical_cpu": self.resource_thresholds.critical_cpu,
                "warning_memory": self.resource_thresholds.warning_memory,
                "critical_memory": self.resource_thresholds.critical_memory,
            },
            
            "accuracy_thresholds": {
                "warning_mape": self.accuracy_thresholds.warning_mape,
                "critical_mape": self.accuracy_thresholds.critical_mape,
                "warning_mae": self.accuracy_thresholds.warning_mae,
                "critical_mae": self.accuracy_thresholds.critical_mae,
            },
            
            "drift_thresholds": {
                "warning_threshold": self.drift_thresholds.warning_threshold,
                "critical_threshold": self.drift_thresholds.critical_threshold,
            },
            
            "enable_alerts": self.enable_alerts,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MonitoringConfig":
        """Cria configuração a partir de dicionário"""
        return cls(
            log_directory=config_dict.get("log_directory", "monitoring_logs"),
            log_frequency_seconds=config_dict.get("log_frequency_seconds", 60),
            latency_thresholds=LatencyThresholds(**config_dict.get("latency_thresholds", {})),
            resource_thresholds=ResourceThresholds(**config_dict.get("resource_thresholds", {})),
            accuracy_thresholds=AccuracyThresholds(**config_dict.get("accuracy_thresholds", {})),
            drift_thresholds=DriftThresholds(**config_dict.get("drift_thresholds", {})),
            enable_alerts=config_dict.get("enable_alerts", True),
        )


# ============================================================================
# Presets de Configuração
# ============================================================================

class MonitoringPresets:
    """Presets predefinidos de configuração"""
    
    @staticmethod
    def development() -> MonitoringConfig:
        """Configuração para ambiente de desenvolvimento"""
        config = MonitoringConfig()
        config.latency_thresholds.critical_p99 = 10000  # Menos rigoroso
        config.resource_thresholds.critical_cpu = 95.0
        config.resource_thresholds.critical_memory = 95.0
        config.enable_alerts = False
        return config
    
    @staticmethod
    def staging() -> MonitoringConfig:
        """Configuração para ambiente de staging"""
        config = MonitoringConfig()
        config.latency_thresholds.critical_p99 = 5000
        config.resource_thresholds.critical_cpu = 90.0
        config.resource_thresholds.critical_memory = 90.0
        return config
    
    @staticmethod
    def production() -> MonitoringConfig:
        """Configuração para ambiente de produção (mais rigorosa)"""
        config = MonitoringConfig()
        config.latency_thresholds.warning_p99 = 1000
        config.latency_thresholds.critical_p99 = 2000
        config.resource_thresholds.warning_cpu = 50.0
        config.resource_thresholds.critical_cpu = 80.0
        config.resource_thresholds.warning_memory = 60.0
        config.resource_thresholds.critical_memory = 85.0
        config.enable_alerts = True
        config.auto_fallback_on_degradation = True
        config.log_retention_days = 90
        return config
    
    @staticmethod
    def high_frequency_trading() -> MonitoringConfig:
        """Configuração para trading de alta frequência (muito rigorosa)"""
        config = MonitoringConfig()
        config.latency_thresholds.warning_p95 = 100
        config.latency_thresholds.critical_p95 = 500
        config.latency_thresholds.warning_p99 = 200
        config.latency_thresholds.critical_p99 = 1000
        config.resource_thresholds.critical_cpu = 70.0
        config.resource_thresholds.critical_memory = 80.0
        config.enable_alerts = True
        config.auto_fallback_on_degradation = True
        return config


# ============================================================================
# Políticas de Alertas
# ============================================================================

class AlertPolicy:
    """Define políticas de como reagir a problemas detectados"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
    
    def get_alert_action(self, severity: AlertSeverity) -> Dict:
        """Retorna ação apropriada para severidade de alerta"""
        actions = {
            AlertSeverity.INFO: {
                "log_only": True,
                "send_alert": False,
                "auto_remediate": False
            },
            AlertSeverity.WARNING: {
                "log_only": False,
                "send_alert": True,
                "auto_remediate": False,
                "notify_managers": False
            },
            AlertSeverity.ERROR: {
                "log_only": False,
                "send_alert": True,
                "auto_remediate": False,
                "notify_managers": True
            },
            AlertSeverity.CRITICAL: {
                "log_only": False,
                "send_alert": True,
                "auto_remediate": self.config.auto_fallback_on_degradation,
                "notify_managers": True,
                "escalate": True
            }
        }
        return actions.get(severity, actions[AlertSeverity.INFO])


# ============================================================================
# Exemplo de Uso
# ============================================================================

if __name__ == "__main__":
    import json
    
    print("\n" + "="*70)
    print("📋 CONFIGURAÇÕES DE MONITORAMENTO DISPONÍVEIS")
    print("="*70 + "\n")
    
    # Development
    print("🔧 DEVELOPMENT")
    print("-" * 70)
    dev_config = MonitoringPresets.development()
    print(json.dumps(dev_config.to_dict(), indent=2))
    
    # Production
    print("\n\n🚀 PRODUCTION")
    print("-" * 70)
    prod_config = MonitoringPresets.production()
    print(json.dumps(prod_config.to_dict(), indent=2))
    
    # High Frequency Trading
    print("\n\n⚡ HIGH FREQUENCY TRADING")
    print("-" * 70)
    hft_config = MonitoringPresets.high_frequency_trading()
    print(json.dumps(hft_config.to_dict(), indent=2))
    
    print("\n" + "="*70 + "\n")
