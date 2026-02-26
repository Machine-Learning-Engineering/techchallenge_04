#!/usr/bin/env python3
"""
Script de Inferência: Carrega modelo treinado e faz previsões
Demonstra como usar o modelo salvo para prever preços de ações
"""

import sys
import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LSTMInferenceEngine:
    """
    Motor de inferência para modelo LSTM treinado
    Carrega modelo pré-treinado e realiza previsões
    """
    
    def __init__(self, model_path: str, scaler_path: str = None):
        """
        Inicializa o motor de inferência
        
        Args:
            model_path: Caminho do modelo treinado (.keras)
            scaler_path: Caminho do arquivo de normalização (pickle)
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = None
        
    def load_model(self) -> bool:
        """
        Carrega o modelo treinado
        
        Returns:
            bool: True se carregado com sucesso
        """
        try:
            if not Path(self.model_path).exists():
                logger.error(f"❌ Modelo não encontrado: {self.model_path}")
                return False
            
            logger.info(f"📂 Carregando modelo de {self.model_path}...")
            self.model = tf.keras.models.load_model(self.model_path)
            
            # Inferir sequence_length desta primeira camada
            input_shape = self.model.input_shape
            self.sequence_length = input_shape[1] if len(input_shape) > 1 else 60
            
            logger.info(f"✅ Modelo carregado com sucesso!")
            logger.info(f"   Input Shape: {self.model.input_shape}")
            logger.info(f"   Sequence Length: {self.sequence_length}")
            logger.info(f"   Total Parameters: {self.model.count_params():,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {str(e)}")
            return False
    
    def load_scaler_from_data(self, csv_path: str, ticker: str = 'AAPL'):
        """
        Carrega e ajusta o scaler com base nos dados históricos
        
        Args:
            csv_path: Caminho do arquivo CSV
            ticker: Ticker da ação
        """
        try:
            logger.info(f"📂 Carregando dados para calibrar normalização...")
            df = pd.read_csv(csv_path)
            df_ticker = df[df['Ticker'] == ticker].copy()
            
            if len(df_ticker) == 0:
                logger.error(f"❌ Nenhum dado encontrado para {ticker}")
                return False
            
            df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])
            df_ticker = df_ticker.sort_values('Date')
            
            prices = df_ticker['Adj Close'].values.reshape(-1, 1)
            self.scaler.fit(prices)
            
            logger.info(f"✅ Normalização calibrada!")
            logger.info(f"   Min: ${prices.min():.2f}")
            logger.info(f"   Max: ${prices.max():.2f}")
            logger.info(f"   Mean: ${prices.mean():.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao calibrar normalização: {str(e)}")
            return False
    
    def predict_single(self, recent_prices: np.ndarray) -> dict:
        """
        Prediz o próximo preço baseado em um histórico recente
        
        Args:
            recent_prices: Array de preços recentes (não normalizados)
            
        Returns:
            dict: Previsão e informações associadas
        """
        if self.model is None:
            logger.error("❌ Modelo não carregado. Execute load_model() primeiro.")
            return None
        
        if len(recent_prices) != self.sequence_length:
            logger.error(f"❌ Esperados {self.sequence_length} preços, recebidos {len(recent_prices)}")
            return None
        
        try:
            # Normalizar
            normalized = self.scaler.transform(recent_prices.reshape(-1, 1))
            
            # Preparar entrada
            X = normalized.reshape(1, self.sequence_length, 1)
            
            # Prever
            prediction_normalized = self.model.predict(X, verbose=0)[0, 0]
            
            # Desnormalizar
            prediction = self.scaler.inverse_transform([[prediction_normalized]])[0, 0]
            
            # Calcular mudança
            current_price = recent_prices[-1]
            change = prediction - current_price
            change_pct = (change / current_price) * 100
            
            return {
                'current_price': float(current_price),
                'predicted_price': float(prediction),
                'change': float(change),
                'change_pct': float(change_pct),
                'confidence': 'HIGH' if abs(change_pct) < 5 else 'MEDIUM' if abs(change_pct) < 10 else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na previsão: {str(e)}")
            return None
    
    def predict_multiple(self, recent_prices: np.ndarray, days: int = 15) -> list:
        """
        Prediz múltiplos dias à frente
        
        Args:
            recent_prices: Array de preços recentes
            days: Número de dias para prever
            
        Returns:
            list: Lista de previsões
        """
        if self.model is None:
            logger.error("❌ Modelo não carregado.")
            return []
        
        predictions = []
        current_sequence = self.scaler.transform(recent_prices.reshape(-1, 1)).flatten()
        
        for _ in range(days):
            # Preparar entrada
            X = current_sequence[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            
            # Prever próximo valor normalizado
            pred_norm = self.model.predict(X, verbose=0)[0, 0]
            
            # Desnormalizar
            pred_value = self.scaler.inverse_transform([[pred_norm]])[0, 0]
            predictions.append(float(pred_value))
            
            # Atualizar sequência
            current_sequence = np.append(current_sequence, pred_norm)
        
        return predictions
    
    def print_model_summary(self):
        """
        Imprime resumo do modelo
        """
        if self.model is None:
            logger.warning("⚠️  Modelo não carregado")
            return
        
        logger.info("\n📋 Resumo do Modelo:")
        self.model.summary()


def example_usage():
    """
    Exemplo de uso do motor de inferência
    """
    logger.info("=" * 70)
    logger.info("EXEMPLO: INFERÊNCIA COM MODELO LSTM")
    logger.info("=" * 70)
    
    # Configurações
    model_path = 'lstm_model_AAPL.keras'
    csv_path = 'data/NASDAQ100_Historical_Data.csv'
    ticker = 'AAPL'
    
    # Criar motor de inferência
    engine = LSTMInferenceEngine(model_path)
    
    # Carregar modelo
    if not engine.load_model():
        return False
    
    # Calibrar normalização
    if not engine.load_scaler_from_data(csv_path, ticker):
        return False
    
    # Carregar últimos dias de preços históricos para usar como entrada
    logger.info(f"\n📊 Carregando histórico recente de {ticker}...")
    df = pd.read_csv(csv_path)
    df_ticker = df[df['Ticker'] == ticker].copy()
    df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])
    df_ticker = df_ticker.sort_values('Date')
    
    recent_prices = df_ticker['Adj Close'].tail(engine.sequence_length).values
    
    logger.info(f"✅ Histórico carregado:")
    logger.info(f"   Preços últimos {engine.sequence_length} dias")
    logger.info(f"   Preço atual: ${recent_prices[-1]:.2f}")
    logger.info(f"   Mínimo: ${recent_prices.min():.2f}")
    logger.info(f"   Máximo: ${recent_prices.max():.2f}")
    
    # Prever próximo dia
    logger.info(f"\n🔮 Previsão para PRÓXIMO DIA:")
    next_day = engine.predict_single(recent_prices)
    if next_day:
        logger.info(f"   Preço Predito: ${next_day['predicted_price']:.2f}")
        logger.info(f"   Mudança: ${next_day['change']:.2f} ({next_day['change_pct']:.2f}%)")
        logger.info(f"   Confiança: {next_day['confidence']}")
    
    # Prever múltiplos dias
    logger.info(f"\n📈 Previsões para PRÓXIMOS 15 DIAS:")
    future_predictions = engine.predict_multiple(recent_prices, days=15)
    current = recent_prices[-1]
    
    for i, pred in enumerate(future_predictions, 1):
        change = pred - current
        change_pct = (change / current) * 100
        direction = "↑" if change > 0 else "↓" if change < 0 else "→"
        logger.info(f"   Dia {i:2d}: ${pred:8.2f} ({direction} ${change:+7.2f} {change_pct:+6.2f}%)")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ EXEMPLO CONCLUÍDO COM SUCESSO!")
    logger.info("=" * 70)
    
    return True


def main():
    """
    Função principal
    """
    try:
        success = example_usage()
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"❌ Erro: {str(e)}")
        logger.exception("Detalhes:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
