"""
INFERÊNCIA COM MONITORAMENTO INTEGRADO
Engine de predição com rastreamento de performance em produção
"""

import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Optional, List
import logging
from model_monitoring import ProductionModelMonitor


class LSTMInferenceEngineWithMonitoring:
    """Engine de inferência LSTM com monitoramento completo"""
    
    def __init__(self, model_path: str, scaler_path: Optional[str] = None, 
                 monitoring_dir: str = "monitoring_logs"):
        """
        Args:
            model_path: Caminho para modelo .keras salvo
            scaler_path: Caminho para scaler (opcional)
            monitoring_dir: Diretório para logs de monitoramento
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.sequence_length = None
        
        # Sistema de monitoramento
        self.monitor = ProductionModelMonitor(log_dir=monitoring_dir)
        self.logger = logging.getLogger("InferenceWithMonitoring")
        
        # Carrega modelo
        self._load_model()
        
        # Carrega scaler
        if scaler_path:
            self._load_scaler()
    
    def _load_model(self):
        """Carrega modelo treinado"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            
            # Extrai sequence_length
            self.sequence_length = self.model.input_shape[1]
            
            self.logger.info(f"✅ Modelo carregado: {self.model_path}")
            self.logger.info(f"   Input Shape: {self.model.input_shape}")
            self.logger.info(f"   Sequence Length: {self.sequence_length}")
            self.logger.info(f"   Total Parameters: {self.model.count_params():,}")
            
            self.monitor.logger.log_event("model_loaded", {
                "model_path": self.model_path,
                "sequence_length": self.sequence_length,
                "parameters": self.model.count_params()
            })
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar modelo: {e}")
            raise
    
    def _load_scaler(self):
        """Carrega scaler para normalização"""
        from sklearn.preprocessing import MinMaxScaler
        import pickle
        
        try:
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.logger.info(f"✅ Scaler carregado: {self.scaler_path}")
        except Exception as e:
            self.logger.warning(f"⚠️  Scaler não disponível: {e}")
    
    def predict_single(self, recent_prices: np.ndarray) -> Dict:
        """
        Predição single-step com monitoramento
        
        Args:
            recent_prices: Array com últimos N preços
            
        Returns:
            Dict com predição, mudança, confiança e métricas
        """
        start_time = time.time()
        
        try:
            # Validação de entrada
            if len(recent_prices) < self.sequence_length:
                recent_prices = np.pad(
                    recent_prices, 
                    (self.sequence_length - len(recent_prices), 0),
                    mode='edge'
                )
            
            recent_prices = recent_prices[-self.sequence_length:]
            
            # Registra dados de entrada para drift detection
            self.monitor.record_input_data(recent_prices)
            
            # Normaliza (se scaler disponível)
            if self.scaler:
                prices_normalized = self.scaler.transform(recent_prices.reshape(-1, 1)).reshape(-1)
            else:
                # Normalização simples [0, 1]
                min_price = np.min(recent_prices)
                max_price = np.max(recent_prices)
                prices_normalized = (recent_prices - min_price) / (max_price - min_price + 1e-10)
            
            # Prepara input para modelo
            X = prices_normalized.reshape(1, self.sequence_length, 1)
            
            # Predição
            prediction_normalized = self.model.predict(X, verbose=0)[0][0]
            
            # Desnormaliza
            if self.scaler:
                # Cria array dummy para desnormalizar
                dummy = np.array([[prediction_normalized]])
                predicted_price = self.scaler.inverse_transform(dummy)[0][0]
            else:
                predicted_price = prediction_normalized * (max_price - min_price) + min_price
            
            # Calcula métricas
            current_price = recent_prices[-1]
            change = predicted_price - current_price
            change_pct = (change / current_price * 100) if current_price != 0 else 0
            
            # Determina confiança baseada na estabilidade da sequência
            price_std = np.std(recent_prices)
            price_mean = np.mean(recent_prices)
            volatility = (price_std / price_mean) if price_mean != 0 else 0
            
            if volatility < 0.01:
                confidence = "HIGH"
            elif volatility < 0.03:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
            
            # Registra predição
            self.monitor.record_prediction(predicted_price, actual_price=None)
            
            # Calcula latência
            latency_ms = (time.time() - start_time) * 1000
            self.monitor.record_inference(latency_ms)
            
            return {
                "current_price": float(current_price),
                "predicted_price": float(predicted_price),
                "change": float(change),
                "change_pct": float(change_pct),
                "confidence": confidence,
                "volatility": float(volatility),
                "latency_ms": float(latency_ms)
            }
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.monitor.logger.log_alert("ERROR", f"Erro em predict_single: {e}")
            self.logger.error(f"❌ Erro na predição: {e}")
            raise
    
    def predict_multiple(self, recent_prices: np.ndarray, days: int = 15) -> List[Dict]:
        """
        Predição multi-step com monitoramento
        
        Args:
            recent_prices: Array com últimos N preços
            days: Número de dias para prever
            
        Returns:
            Lista de predições para cada dia
        """
        predictions = []
        current_sequence = recent_prices[-self.sequence_length:].copy()
        
        for day in range(1, days + 1):
            # Predição para próximo dia
            pred_dict = self._predict_internal(current_sequence)
            pred_dict["day"] = day
            predictions.append(pred_dict)
            
            # Atualiza sequência para próxima iteração
            predicted_price = pred_dict["predicted_price"]
            current_sequence = np.append(current_sequence, predicted_price)[1:]
        
        return predictions
    
    def _predict_internal(self, prices: np.ndarray) -> Dict:
        """Predição interna para multi-step"""
        # Normaliza
        if self.scaler:
            prices_normalized = self.scaler.transform(prices.reshape(-1, 1)).reshape(-1)
        else:
            min_p = np.min(prices)
            max_p = np.max(prices)
            prices_normalized = (prices - min_p) / (max_p - min_p + 1e-10)
        
        X = prices_normalized.reshape(1, self.sequence_length, 1)
        prediction_normalized = self.model.predict(X, verbose=0)[0][0]
        
        # Desnormaliza
        if self.scaler:
            dummy = np.array([[prediction_normalized]])
            predicted_price = self.scaler.inverse_transform(dummy)[0][0]
        else:
            predicted_price = prediction_normalized * (max_p - min_p) + min_p
        
        current_price = prices[-1]
        change = predicted_price - current_price
        change_pct = (change / current_price * 100) if current_price != 0 else 0
        
        return {
            "current_price": float(current_price),
            "predicted_price": float(predicted_price),
            "change": float(change),
            "change_pct": float(change_pct)
        }
    
    def get_monitoring_report(self) -> Dict:
        """Retorna relatório de monitoramento"""
        return self.monitor.generate_report()
    
    def print_monitoring_summary(self):
        """Imprime resumo de monitoramento"""
        self.monitor.print_summary()
    
    def export_monitoring_report(self, filepath: str):
        """Exporta relatório de monitoramento"""
        self.monitor.export_report(filepath)


# ============================================================================
# EXEMPLO DE USO COM MONITORAMENTO
# ============================================================================

if __name__ == "__main__":
    import pandas as pd
    
    print("\n" + "="*70)
    print("🚀 INFERÊNCIA COM MONITORAMENTO INTEGRADO")
    print("="*70 + "\n")
    
    # Carrega dados históricos
    data_path = Path("data/NASDAQ100_Historical_Data.csv")
    if not data_path.exists():
        print(f"❌ Arquivo não encontrado: {data_path}")
        exit(1)
    
    print("📂 Carregando dados históricos...")
    df = pd.read_csv(data_path)
    df = df[df['Ticker'] == 'AAPL'].sort_values('Date').reset_index(drop=True)
    print(f"✅ {len(df)} registros de AAPL carregados\n")
    
    # Inicializa engine com monitoramento
    print("🔧 Inicializando engine de inferência com monitoramento...")
    engine = LSTMInferenceEngineWithMonitoring(
        model_path="lstm_model_AAPL.keras",
        monitoring_dir="monitoring_logs"
    )
    print("✅ Engine inicializado\n")
    
    # Obtém últimos 60 preços
    recent_prices = df['Close'].tail(60).values
    
    print("="*70)
    print("📊 PREDIÇÃO SINGLE-STEP COM MONITORAMENTO")
    print("="*70)
    pred = engine.predict_single(recent_prices)
    print(f"\n💰 Preço Atual: ${pred['current_price']:.2f}")
    print(f"🔮 Preço Predito: ${pred['predicted_price']:.2f}")
    print(f"📈 Mudança: ${pred['change']:+.2f} ({pred['change_pct']:+.2f}%)")
    print(f"🎯 Confiança: {pred['confidence']}")
    print(f"📉 Volatilidade: {pred['volatility']:.4f}")
    print(f"⚡ Latência: {pred['latency_ms']:.2f}ms")
    
    print("\n" + "="*70)
    print("📈 PREDIÇÃO MULTI-STEP (15 DIAS)")
    print("="*70 + "\n")
    predictions = engine.predict_multiple(recent_prices, days=15)
    
    for pred in predictions:
        print(f"Dia {pred['day']:2d}: ${pred['predicted_price']:8.2f} "
              f"({pred['change_pct']:+6.2f}%)")
    
    print("\n" + "="*70)
    print("💻 RELATÓRIO DE MONITORAMENTO")
    print("="*70)
    engine.print_monitoring_summary()
    
    # Exporta relatório
    report_file = "inference_monitoring_report.json"
    engine.export_monitoring_report(report_file)
    print(f"📄 Relatório exportado: {report_file}")
