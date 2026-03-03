"""
CONFIGURAÇÃO CENTRALIZADA DE LOGGING
Setup parametrizável via variáveis de ambiente
"""

import logging
import logging.handlers
from pathlib import Path
import os
import sys
from datetime import datetime


class LoggingConfig:
    """Gerencia a configuração de logging para toda a aplicação"""
    
    # Níveis de logging suportados
    LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    # Formato padrão com debug detalhado
    FORMAT_DEBUG = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
    
    # Formato padrão para info/warning/error
    FORMAT_STANDARD = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Formato minimalista para produção
    FORMAT_MINIMAL = '%(asctime)s - %(levelname)s - %(message)s'
    
    def __init__(self):
        """Inicializa a configuração de logging"""
        self.log_level = self._get_log_level_from_env()
        self.log_dir = self._get_log_dir_from_env()
        self.enable_file_logging = self._get_file_logging_from_env()
        self.enable_console_logging = self._get_console_logging_from_env()
        self._setup_logging()
    
    @staticmethod
    def _get_log_level_from_env() -> int:
        """Obtém log level das variáveis de ambiente"""
        level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
        return LoggingConfig.LOG_LEVELS.get(level_str, logging.INFO)
    
    @staticmethod
    def _get_log_dir_from_env() -> Path:
        """Obtém diretório de logs das variáveis de ambiente"""
        log_dir = os.getenv('LOG_DIR', 'logs')
        log_path = Path(log_dir)
        
        try:
            # Criar diretório se não existir
            log_path.mkdir(parents=True, exist_ok=True)
            # Garantir permissões corretas
            log_path.chmod(0o755)
        except PermissionError as e:
            # Se não tiver permissão, tentar usar /tmp como fallback
            log_path = Path('/tmp/lstm-logs')
            log_path.mkdir(parents=True, exist_ok=True)
            log_path.chmod(0o755)
            print(f"⚠️  Sem permissão para escrever em {log_dir}, usando /tmp como fallback", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  Erro ao criar diretório de logs: {e}", file=sys.stderr)
            # Usar /tmp como último recurso
            log_path = Path('/tmp/lstm-logs')
            log_path.mkdir(parents=True, exist_ok=True)
        
        return log_path
    
    @staticmethod
    def _get_file_logging_from_env() -> bool:
        """Define se deve usar file logging"""
        return os.getenv('ENABLE_FILE_LOGGING', 'true').lower() == 'true'
    
    @staticmethod
    def _get_console_logging_from_env() -> bool:
        """Define se deve usar console logging"""
        return os.getenv('ENABLE_CONSOLE_LOGGING', 'true').lower() == 'true'
    
    def _setup_logging(self):
        """Configura o logging para toda a aplicação"""
        # Obter o root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Remover handlers existentes para evitar duplicação
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Escolher formato baseado no log level
        if self.log_level == logging.DEBUG:
            fmt = LoggingConfig.FORMAT_DEBUG
        else:
            fmt = LoggingConfig.FORMAT_STANDARD
        
        formatter = logging.Formatter(fmt)
        
        # Handler para console
        if self.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Handler para arquivo
        if self.enable_file_logging:
            try:
                log_file = self.log_dir / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                # Garantir que o arquivo pode ser criado
                log_file.parent.mkdir(parents=True, exist_ok=True)
                log_file.parent.chmod(0o755)
                
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(self.log_level)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
            except PermissionError:
                # Se não tiver permissão, tentar usar /tmp
                try:
                    log_file = Path('/tmp/lstm-logs') / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                    log_file.parent.mkdir(parents=True, exist_ok=True)
                    file_handler = logging.FileHandler(log_file, encoding='utf-8')
                    file_handler.setLevel(self.log_level)
                    file_handler.setFormatter(formatter)
                    root_logger.addHandler(file_handler)
                    console = logging.getLogger()
                    console.warning(f"⚠️  Sem permissão no LOG_DIR, usando {log_file.parent} como fallback")
                except Exception as e:
                    # Se tudo falhar, apenas usar console
                    console = logging.getLogger()
                    console.warning(f"⚠️  Não foi possível criar arquivo de log: {e}")
            except Exception as e:
                # Qualquer outro erro, apenas usar console
                console = logging.getLogger()
                console.warning(f"⚠️  Erro ao criar arquivo de log: {e}")
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Obtém um logger configurado
        
        Args:
            name: Nome do logger (geralmente __name__)
            
        Returns:
            Logger configurado
        """
        return logging.getLogger(name)
    
    @staticmethod
    def set_log_level(level: str):
        """
        Altera o log level dynamicamente
        
        Args:
            level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        """
        level_int = LoggingConfig.LOG_LEVELS.get(level.upper(), logging.INFO)
        root_logger = logging.getLogger()
        root_logger.setLevel(level_int)
        
        for handler in root_logger.handlers:
            handler.setLevel(level_int)
        
        logger = logging.getLogger(__name__)
        logger.info(f"📊 Log level alterado para: {level}")


# Instancia global de configuração
_config = LoggingConfig()

# Função de conveniência
def get_logger(name: str) -> logging.Logger:
    """Atalho para obter um logger"""
    return LoggingConfig.get_logger(name)


def init_logging():
    """Inicializa o sistema de logging"""
    logger = get_logger('Logging')
    level_name = os.getenv('LOG_LEVEL', 'INFO').upper()
    logger.info(f"✅ Sistema de logging inicializado - Level: {level_name}")
