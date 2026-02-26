#!/usr/bin/env python3
"""
Modelo de Deep Learning com LSTM para previsão de preços de ações NASDAQ-100
Captura padrões temporais nos dados históricos de preços
"""

import sys
import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Keras/TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suprimir warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LSTMStockPricePredictor:
    """
    Modelo LSTM para previsão de preços de ações
    """
    
    def __init__(self, sequence_length: int = 60, ticker: str = 'AAPL', random_seed: int = 42):
        """
        Inicializa o preditor LSTM
        
        Args:
            sequence_length: Número de dias anteriores para prever próximo dia
            ticker: Ticker da ação para treinar
        """
        self.sequence_length = sequence_length
        self.ticker = ticker
        self.random_seed = random_seed
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_predictions = None
        self.test_predictions = None
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Carrega os dados do CSV
        
        Args:
            csv_path: Caminho do arquivo CSV
            
        Returns:
            DataFrame filtrado para o ticker especificado
        """
        logger.info(f"📂 Carregando dados de {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Filtrar por ticker
        df_ticker = df[df['Ticker'] == self.ticker].copy()
        
        logger.info(f"✅ Dados carregados: {len(df_ticker)} registros para {self.ticker}")
        
        # Ordenar por data
        df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])
        df_ticker = df_ticker.sort_values('Date').reset_index(drop=True)
        
        logger.info(f"   📅 Período: {df_ticker['Date'].min().date()} até {df_ticker['Date'].max().date()}")
        logger.info(f"   💰 Price Range: ${df_ticker['Close'].min():.2f} - ${df_ticker['Close'].max():.2f}")
        
        return df_ticker
    
    def prepare_data(self, df: pd.DataFrame, train_size: float = 0.7, val_size: float = 0.1):
        """
        Prepara os dados para o modelo LSTM
        
        Args:
            df: DataFrame com os dados
            train_size: Proporção de dados para treinamento (0.7 = 70%)
            val_size: Proporção de dados para validação (0.1 = 10%)
        """
        logger.info(f"\n📊 Preparando dados...")
        
        # Usar a coluna de preço de fechamento ajustado
        data = df['Adj Close'].values.reshape(-1, 1)
        
        # Dividir em treino, validação e teste (split temporal)
        train_end = int(len(data) * train_size)
        val_end = int(len(data) * (train_size + val_size))
        
        train_raw = data[:train_end]
        val_raw = data[train_end:val_end]
        test_raw = data[val_end:]
        
        # Normalizar com base apenas no treino para evitar vazamento
        logger.info("   Normalizando dados com MinMaxScaler (fit no treino)...")
        self.scaler.fit(train_raw)
        self.train_data = self.scaler.transform(train_raw)
        self.val_data = self.scaler.transform(val_raw)
        self.test_data = self.scaler.transform(test_raw)
        
        logger.info(f"   📈 Dados de treinamento: {len(self.train_data)} amostras")
        logger.info(f"   🧪 Dados de validação: {len(self.val_data)} amostras")
        logger.info(f"   📉 Dados de teste: {len(self.test_data)} amostras")
        
    def create_sequences(self, data: np.ndarray) -> tuple:
        """
        Cria sequências temporais para o LSTM
        
        Args:
            data: Dados normalizados
            
        Returns:
            X, y - Sequências de entrada e saída
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_model(
        self,
        input_shape: tuple,
        lstm_units: int = 50,
        dense_units: int = 25,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Constrói o modelo LSTM
        
        Args:
            input_shape: Forma dos dados de entrada
            lstm_units: Número de unidades nas camadas LSTM
            dense_units: Número de unidades na camada densa
            dropout_rate: Taxa de dropout
            learning_rate: Taxa de aprendizado
        """
        logger.info(f"\n🔨 Construindo modelo LSTM...")
        
        self.model = Sequential([
            # Primeira camada LSTM com 50 unidades e retorno de sequências
            LSTM(
                units=lstm_units,
                return_sequences=True,
                input_shape=input_shape,
                activation='relu'
            ),
            Dropout(dropout_rate),
            
            # Segunda camada LSTM com 50 unidades
            LSTM(units=lstm_units, return_sequences=False, activation='relu'),
            Dropout(dropout_rate),
            
            # Camadas densas
            Dense(units=dense_units, activation='relu'),
            Dropout(dropout_rate),
            
            # Camada de saída
            Dense(units=1)
        ])
        
        # Compilar o modelo
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        logger.info(f"✅ Modelo compilado!")
        logger.info(f"\n📋 Arquitetura do modelo:")
        self.model.summary()
        
    def _train_model(self, X_train, y_train, X_val, y_val, epochs: int, batch_size: int):
        """
        Treina o modelo com dados de treino e validação explícitos
        """
        logger.info(f"\n🚀 Iniciando treinamento...")
        
        # Callbacks para melhorar o treinamento
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        logger.info(f"   Formato X_train: {X_train.shape}")
        logger.info(f"   Formato y_train: {y_train.shape}")
        logger.info(f"   Formato X_val: {X_val.shape}")
        logger.info(f"   Formato y_val: {y_val.shape}")
        
        # Treinar
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"✅ Treinamento concluído!")
        
        return history

    def train(self, epochs: int = 50, batch_size: int = 32):
        """
        Treina o modelo LSTM usando treino e validação já preparados
        """
        if self.train_data is None or self.val_data is None:
            raise ValueError("Dados de treino/validação não preparados. Execute prepare_data primeiro.")
        
        X_train, y_train = self.create_sequences(self.train_data)
        X_val, y_val = self.create_sequences(self.val_data)
        
        return self._train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

    def tune_hyperparameters(self, tune_fraction: float = 0.5):
        """
        Ajusta hiperparâmetros usando apenas uma parte dos dados de treino/validação
        """
        if self.train_data is None or self.val_data is None:
            raise ValueError("Dados de treino/validação não preparados. Execute prepare_data primeiro.")
        
        logger.info("\n🧪 Iniciando ajuste de hiperparâmetros...")
        logger.info(f"   Usando {int(tune_fraction * 100)}% dos dados para tuning")
        
        tune_train_size = max(int(len(self.train_data) * tune_fraction), self.sequence_length + 5)
        tune_val_size = max(int(len(self.val_data) * tune_fraction), self.sequence_length + 5)
        
        tune_train = self.train_data[:tune_train_size]
        tune_val = self.val_data[:tune_val_size]
        
        search_space = [
            {"sequence_length": 60, "lstm_units": 50, "dense_units": 25, "dropout_rate": 0.2, "learning_rate": 0.001, "batch_size": 32},
            {"sequence_length": 60, "lstm_units": 64, "dense_units": 32, "dropout_rate": 0.1, "learning_rate": 0.001, "batch_size": 32},
            {"sequence_length": 60, "lstm_units": 64, "dense_units": 32, "dropout_rate": 0.2, "learning_rate": 0.0005, "batch_size": 32},
            {"sequence_length": 90, "lstm_units": 64, "dense_units": 32, "dropout_rate": 0.2, "learning_rate": 0.0005, "batch_size": 64},
            {"sequence_length": 30, "lstm_units": 64, "dense_units": 32, "dropout_rate": 0.2, "learning_rate": 0.001, "batch_size": 32}
        ]
        
        best_config = None
        best_val_loss = float("inf")
        
        for idx, config in enumerate(search_space, 1):
            logger.info("\n" + "-" * 60)
            logger.info(f"Teste {idx}/{len(search_space)} - Config: {config}")
            
            self.sequence_length = config["sequence_length"]
            
            X_train, y_train = self.create_sequences(tune_train)
            X_val, y_val = self.create_sequences(tune_val)
            
            if len(X_train) == 0 or len(X_val) == 0:
                logger.warning("Config ignorada por tamanho insuficiente de sequência")
                continue
            
            self.build_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                lstm_units=config["lstm_units"],
                dense_units=config["dense_units"],
                dropout_rate=config["dropout_rate"],
                learning_rate=config["learning_rate"]
            )
            
            history = self._train_model(
                X_train,
                y_train,
                X_val,
                y_val,
                epochs=25,
                batch_size=config["batch_size"]
            )
            
            val_loss = min(history.history.get("val_loss", [float("inf")]))
            logger.info(f"   Melhor val_loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config = config
        
        if best_config is None:
            raise RuntimeError("Nenhuma configuração válida foi encontrada no tuning.")
        
        logger.info("\n✅ Melhor configuração encontrada:")
        logger.info(f"   Config: {best_config}")
        logger.info(f"   val_loss: {best_val_loss:.6f}")
        
        return best_config
    
    def predict(self):
        """
        Faz predições nos dados de treino e teste
        """
        logger.info(f"\n🔮 Fazendo predições...")
        
        # Predições no conjunto de treinamento
        X_train, _ = self.create_sequences(self.train_data)
        self.train_predictions = self.model.predict(X_train, verbose=0)
        
        # Predições no conjunto de validação
        if self.val_data is not None:
            X_val, self.y_val = self.create_sequences(self.val_data)
            self.val_predictions = self.model.predict(X_val, verbose=0)
        else:
            self.y_val = None
            self.val_predictions = None

        # Predições no conjunto de teste
        X_test, self.y_test = self.create_sequences(self.test_data)
        self.test_predictions = self.model.predict(X_test, verbose=0)
        
        logger.info(f"✅ Predições realizadas!")
        logger.info(f"   Predições de treino: {self.train_predictions.shape}")
        if self.val_predictions is not None:
            logger.info(f"   Predições de validação: {self.val_predictions.shape}")
        logger.info(f"   Predições de teste: {self.test_predictions.shape}")
    
    def _compute_metrics(self, y_true, y_pred) -> dict:
        """
        Calcula métricas de avaliação
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "mape": mape}

    def evaluate(self):
        """
        Avalia o desempenho do modelo
        """
        logger.info(f"\n📊 Avaliando modelo...")
        
        # Desnormalizar predições e valores reais para comparação
        train_predictions_actual = self.scaler.inverse_transform(self.train_predictions)
        y_train_actual = self.scaler.inverse_transform(self.train_data[self.sequence_length:])
        
        if self.val_predictions is not None and self.y_val is not None:
            val_predictions_actual = self.scaler.inverse_transform(self.val_predictions)
            y_val_actual = self.scaler.inverse_transform(self.y_val)
        else:
            val_predictions_actual = None
            y_val_actual = None

        test_predictions_actual = self.scaler.inverse_transform(self.test_predictions)
        y_test_actual = self.scaler.inverse_transform(self.y_test)

        train_metrics = self._compute_metrics(y_train_actual, train_predictions_actual)
        test_metrics = self._compute_metrics(y_test_actual, test_predictions_actual)

        logger.info(f"\n📈 Métricas de TREINAMENTO:")
        logger.info(f"   RMSE: ${train_metrics['rmse']:.4f}")
        logger.info(f"   MAE:  ${train_metrics['mae']:.4f}")
        logger.info(f"   MSE:  ${train_metrics['mse']:.4f}")
        logger.info(f"   R²:   {train_metrics['r2']:.4f}")
        logger.info(f"   MAPE: {train_metrics['mape']:.2f}%")

        if val_predictions_actual is not None and y_val_actual is not None:
            val_metrics = self._compute_metrics(y_val_actual, val_predictions_actual)
            logger.info(f"\n🧪 Métricas de VALIDAÇÃO:")
            logger.info(f"   RMSE: ${val_metrics['rmse']:.4f}")
            logger.info(f"   MAE:  ${val_metrics['mae']:.4f}")
            logger.info(f"   MSE:  ${val_metrics['mse']:.4f}")
            logger.info(f"   R²:   {val_metrics['r2']:.4f}")
            logger.info(f"   MAPE: {val_metrics['mape']:.2f}%")
        else:
            val_metrics = None

        logger.info(f"\n📉 Métricas de TESTE:")
        logger.info(f"   RMSE: ${test_metrics['rmse']:.4f}")
        logger.info(f"   MAE:  ${test_metrics['mae']:.4f}")
        logger.info(f"   MSE:  ${test_metrics['mse']:.4f}")
        logger.info(f"   R²:   {test_metrics['r2']:.4f}")
        logger.info(f"   MAPE: {test_metrics['mape']:.2f}%")

        return {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }
    
    def save_model(self, filepath: str = 'lstm_stock_model.keras', format: str = 'keras'):
        """
        Salva o modelo treinado em diferentes formatos
        
        Args:
            filepath: Caminho para salvar o modelo (sem extensão)
            format: Formato de salvamento ('keras', 'savedmodel', or 'all')
        """
        logger.info(f"\n💾 Salvando modelo...")
        
        if format in ['keras', 'all']:
            keras_path = filepath if filepath.endswith('.keras') else f"{filepath}.keras"
            logger.info(f"   Salvando em Keras (.keras): {keras_path}")
            self.model.save(keras_path)
            logger.info(f"   ✅ Keras format: {Path(keras_path).stat().st_size / 1024:.1f} KB")
        
        if format in ['savedmodel', 'all']:
            savedmodel_dir = filepath if not filepath.endswith('.keras') else filepath.replace('.keras', '')
            logger.info(f"   Salvando em TensorFlow SavedModel: {savedmodel_dir}")
            self.model.save(savedmodel_dir)
            logger.info(f"   ✅ SavedModel format salvo")
        
        try:
            if format in ['onnx', 'all']:
                import tf2onnx
                import onnx
                onnx_path = filepath if filepath.endswith('.onnx') else f"{filepath}.onnx"
                logger.info(f"   Salvando em ONNX (.onnx): {onnx_path}")
                spec = (tf.TensorSpec((None, self.sequence_length, 1), tf.float32, name="input"),)
                output_path = tf2onnx.convert.from_keras(self.model, input_signature=spec, output_path=onnx_path)
                logger.info(f"   ✅ ONNX format: {Path(onnx_path).stat().st_size / 1024:.1f} KB")
        except ImportError:
            logger.warning("   ⚠️  tf2onnx não disponível para formato ONNX")
        
        logger.info(f"✅ Modelo salvo com sucesso!")
    
    def plot_results(self, output_dir: str = '.'):
        """
        Plota os resultados das predições
        
        Args:
            output_dir: Diretório para salvar gráficos
        """
        logger.info(f"\n📊 Gerando gráficos...")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Desnormalizar para visualização
        test_predictions_actual = self.scaler.inverse_transform(self.test_predictions)
        y_test_actual = self.scaler.inverse_transform(self.y_test)
        
        # Gráfico 1: Previsões vs Valores Reais
        plt.figure(figsize=(14, 5))
        plt.plot(y_test_actual, label='Preço Real', linewidth=2)
        plt.plot(test_predictions_actual, label='Previsão LSTM', linewidth=2, alpha=0.8)
        plt.xlabel('Dias')
        plt.ylabel('Preço ($)')
        plt.title(f'{self.ticker} - Previsão LSTM vs Preço Real (Conjunto de Teste)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/lstm_predictions_{self.ticker}.png', dpi=300)
        logger.info(f"   ✅ Gráfico salvo: lstm_predictions_{self.ticker}.png")
        
        # Gráfico 2: Distribuição de Resíduos
        residuals = y_test_actual - test_predictions_actual
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Resíduo ($)')
        plt.ylabel('Frequência')
        plt.title('Distribuição de Resíduos')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(residuals, marker='o', linestyle='-', markersize=3, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('Dias')
        plt.ylabel('Resíduo ($)')
        plt.title('Resíduos ao Longo do Tempo')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/lstm_residuals_{self.ticker}.png', dpi=300)
        logger.info(f"   ✅ Gráfico salvo: lstm_residuals_{self.ticker}.png")
        
        plt.close('all')
    
    def generate_evaluation_report(self, metrics: dict, output_file: str = 'model_evaluation_report.txt'):
        """
        Gera relatório de avaliação do modelo
        
        Args:
            metrics: Dicionário com métricas dos conjuntos train/val/test
            output_file: Arquivo para salvar o relatório
        """
        logger.info(f"\n📄 Gerando relatório de avaliação...")
        
        report = f"""
╔═════════════════════════════════════════════════════════════════════════════╗
║                    RELATÓRIO DE AVALIAÇÃO - MODELO LSTM                    ║
║                    Previsão de Preços NASDAQ-100                           ║
╚═════════════════════════════════════════════════════════════════════════════╝

📊 INFORMAÇÕES DO MODELO
───────────────────────────────────────────────────────────────────────────────
Ticker:                    {self.ticker}
Sequence Length:           {self.sequence_length}
Data de Geração:           {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

📈 MÉTRICAS DE TREINAMENTO
───────────────────────────────────────────────────────────────────────────────
RMSE (Root Mean Squared Error): ${metrics['train']['rmse']:.4f}
MAE (Mean Absolute Error):      ${metrics['train']['mae']:.4f}
MSE (Mean Squared Error):       ${metrics['train']['mse']:.4f}
R² Score:                       {metrics['train']['r2']:.4f} (Explica {metrics['train']['r2']*100:.2f}% da variância)
MAPE (Mean Absolute Percentage Error): {metrics['train']['mape']:.2f}%

🧪 MÉTRICAS DE VALIDAÇÃO
───────────────────────────────────────────────────────────────────────────────
"""
        
        if metrics['val'] is not None:
            report += f"""
RMSE:                       ${metrics['val']['rmse']:.4f}
MAE:                        ${metrics['val']['mae']:.4f}
MSE:                        ${metrics['val']['mse']:.4f}
R² Score:                   {metrics['val']['r2']:.4f} (Explica {metrics['val']['r2']*100:.2f}% da variância)
MAPE:                       {metrics['val']['mape']:.2f}%
"""
        else:
            report += "Não disponível\n"
        
        report += f"""
📉 MÉTRICAS DE TESTE (Avaliação Final)
───────────────────────────────────────────────────────────────────────────────
RMSE:                       ${metrics['test']['rmse']:.4f}
MAE:                        ${metrics['test']['mae']:.4f}
MSE:                        ${metrics['test']['mse']:.4f}
R² Score:                   {metrics['test']['r2']:.4f} (Explica {metrics['test']['r2']*100:.2f}% da variância)
MAPE:                       {metrics['test']['mape']:.2f}%

📊 INTERPRETAÇÃO DAS MÉTRICAS
───────────────────────────────────────────────────────────────────────────────
RMSE (Root Mean Squared Error):
  - Mede o erro quadrático médio entre previsões e valores reais
  - Penaliza mais erros maiores
  - Mesma unidade que a variável-alvo (dólares)
  - Valor menor = melhor

MAE (Mean Absolute Error):
  - Erro médio absoluto entre previsões e valores reais
  - Fácil de interpretar: em média, previsões se desviam MAE dólares
  - Valor menor = melhor

MAPE (Mean Absolute Percentage Error):
  - Erro percentual médio em relação aos valores reais
  - Permite comparação entre diferentes escalas de preço
  - < 5%: Excelente
  - 5-10%: Muito bom
  - 10-20%: Bom
  - > 20%: Precisa melhorias

R² Score (Coefficient of Determination):
  - Proporção da variância explicada pelo modelo
  - Varia de 0 a 1 (1 = perfeito)
  - > 0.9: Excelente
  - 0.8-0.9: Muito bom
  - 0.7-0.8: Bom
  - < 0.7: Precisa melhorias

🎯 CONCLUSÃO
───────────────────────────────────────────────────────────────────────────────
"""
        
        # Análise automática
        test_r2 = metrics['test']['r2']
        test_mape = metrics['test']['mape']
        
        if test_r2 > 0.9 and test_mape < 5:
            conclusion = "✅ EXCELENTE: Modelo pronto para uso em produção"
        elif test_r2 > 0.85 and test_mape < 10:
            conclusion = "✅ MUITO BOM: Modelo com ótimo desempenho"
        elif test_r2 > 0.75 and test_mape < 15:
            conclusion = "✅ BOM: Modelo viável com possíveis melhorias"
        else:
            conclusion = "⚠️  PRECISA MELHORIAS: Considere ajustar hiperparâmetros"
        
        report += conclusion
        report += f"""

O modelo alcançou R² de {test_r2:.4f} e MAPE de {test_mape:.2f}% no conjunto de teste,
indicando que explica {test_r2*100:.2f}% da variância nos preços e tem um erro
médio de {test_mape:.2f}% nas previsões.

═════════════════════════════════════════════════════════════════════════════════
Relatório Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
═════════════════════════════════════════════════════════════════════════════════
"""
        
        # Salvar relatório
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"✅ Relatório salvo em: {output_file}")
        print(report)


def main():
    """Função principal"""
    logger.info("=" * 70)
    logger.info("MODELO LSTM PARA PREVISÃO DE PREÇOS - NASDAQ-100")
    logger.info("=" * 70)
    
    try:
        # Configurações
        csv_path = 'data/NASDAQ100_Historical_Data.csv'
        ticker = 'AAPL'
        random_seed = 42
        train_size = 0.7
        val_size = 0.1
        tune_fraction = 0.5
        final_epochs = 60
        
        # Criar preditor
        tf.keras.utils.set_random_seed(random_seed)
        predictor = LSTMStockPricePredictor(
            sequence_length=60,
            ticker=ticker,
            random_seed=random_seed
        )
        
        # Carregar dados
        df = predictor.load_data(csv_path)
        
        # Preparar dados (split temporal)
        predictor.prepare_data(df, train_size=train_size, val_size=val_size)
        
        # Ajustar hiperparâmetros usando parte dos dados
        best_config = predictor.tune_hyperparameters(tune_fraction=tune_fraction)
        
        # Construir modelo com a melhor configuração
        predictor.sequence_length = best_config["sequence_length"]
        X_train, _ = predictor.create_sequences(predictor.train_data)
        predictor.build_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            lstm_units=best_config["lstm_units"],
            dense_units=best_config["dense_units"],
            dropout_rate=best_config["dropout_rate"],
            learning_rate=best_config["learning_rate"]
        )
        
        # Treinar modelo com todos os dados de treino/validação
        history = predictor.train(epochs=final_epochs, batch_size=best_config["batch_size"])
        
        # Fazer predições
        predictor.predict()
        
        # Avaliar modelo
        metrics = predictor.evaluate()
        
        # Salvar modelo em múltiplos formatos
        predictor.save_model(f'lstm_model_{ticker}', format='keras')
        
        # Plotar resultados
        predictor.plot_results('.')
        
        # Gerar relatório de avaliação
        predictor.generate_evaluation_report(metrics, f'model_evaluation_{ticker}.txt')
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ PROCESSO CONCLUÍDO COM SUCESSO!")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Erro: {str(e)}")
        logger.exception("Detalhes:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
