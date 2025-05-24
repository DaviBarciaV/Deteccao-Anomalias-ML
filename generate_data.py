import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_transactions(num_transactions=1000, anomaly_ratio=0.05):
    """
    Gera um DataFrame de transações financeiras sintéticas com anomalias injetadas.
    """
    np.random.seed(42) # Para reprodutibilidade

    data = []
    num_anomalies = int(num_transactions * anomaly_ratio)
    num_normal = num_transactions - num_anomalies

    # --- Gerar Transações Normais ---
    for i in range(num_normal):
        transaction_id = f"txn_normal_{i:04d}"
        # Montantes normais entre 10 e 500
        amount = np.random.uniform(10, 500)
        # Horas normais de transação (8h - 22h)
        timestamp = datetime(2023, 1, 1, np.random.randint(8, 23), np.random.randint(0, 60), np.random.randint(0, 60))
        merchant_category = np.random.choice(['Eletrônicos', 'Supermercado', 'Restaurante', 'Vestuário', 'Viagem'])
        user_id = f"user_{np.random.randint(1, 101):03d}" # 100 usuários
        location = np.random.choice(['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Curitiba'])
        # Feature adicional: frequência de transações do usuário nas últimas 24h (simulado)
        user_tx_freq_24h = np.random.randint(1, 5)
        # Feature adicional: tempo desde a última transação do usuário (em minutos, simulado)
        time_since_last_tx_user_min = np.random.randint(5, 360)

        data.append([
            transaction_id, timestamp, amount, merchant_category, user_id, location,
            user_tx_freq_24h, time_since_last_tx_user_min, 0 # 0 para normal
        ])

    # --- Gerar Transações Anômalas ---
    for i in range(num_anomalies):
        transaction_id = f"txn_anomaly_{i:04d}"
        anomaly_type = np.random.choice(['high_amount', 'unusual_time', 'high_frequency', 'unusual_location_category_combo'])

        if anomaly_type == 'high_amount':
            # Montantes anômalos (muito altos)
            amount = np.random.uniform(2000, 10000)
            timestamp = datetime(2023, 1, 1, np.random.randint(8, 23), np.random.randint(0, 60), np.random.randint(0, 60))
            merchant_category = np.random.choice(['Eletrônicos', 'Supermercado', 'Restaurante', 'Vestuário', 'Viagem'])
            location = np.random.choice(['São Paulo', 'Rio de Janeiro'])
            user_tx_freq_24h = np.random.randint(1, 5)
            time_since_last_tx_user_min = np.random.randint(5, 360)
        elif anomaly_type == 'unusual_time':
            amount = np.random.uniform(50, 300)
            # Horas anômalas (madrugada)
            timestamp = datetime(2023, 1, 1, np.random.randint(0, 6), np.random.randint(0, 60), np.random.randint(0, 60))
            merchant_category = 'Restaurante' # Restaurantes geralmente não operam de madrugada
            location = 'São Paulo'
            user_tx_freq_24h = np.random.randint(1, 3)
            time_since_last_tx_user_min = np.random.randint(60, 720) # Mais tempo desde a última
        elif anomaly_type == 'high_frequency':
            amount = np.random.uniform(20, 100) # Valores menores, mas alta frequência
            timestamp = datetime(2023, 1, 1, np.random.randint(10, 18), np.random.randint(0, 60), np.random.randint(0, 60))
            merchant_category = np.random.choice(['Supermercado', 'Vestuário'])
            location = 'Curitiba'
            user_tx_freq_24h = np.random.randint(10, 25) # Frequência muito alta
            time_since_last_tx_user_min = np.random.randint(1, 10) # Tempo muito curto desde a última
        else: # unusual_location_category_combo
            amount = np.random.uniform(100, 600)
            timestamp = datetime(2023, 1, 1, np.random.randint(10, 18), np.random.randint(0, 60), np.random.randint(0, 60))
            merchant_category = 'Viagem' # Compra de viagem
            location = 'Cidade Pequena Distante' # Localização incomum para este tipo de compra frequente
            user_tx_freq_24h = np.random.randint(1, 3)
            time_since_last_tx_user_min = np.random.randint(120, 1000)


        user_id = f"user_{np.random.randint(201, 251):03d}" # Usuários diferentes para algumas anomalias

        data.append([
            transaction_id, timestamp, amount, merchant_category, user_id, location,
            user_tx_freq_24h, time_since_last_tx_user_min, 1 # 1 para anomalia
        ])

    df = pd.DataFrame(data, columns=[
        'TransactionID', 'Timestamp', 'Amount', 'MerchantCategory', 'UserID', 'Location',
        'UserTxFreq24h', 'TimeSinceLastTxUserMin', 'IsKnownAnomaly_GroundTruth' # Ground truth para avaliação (opcional)
    ])

    # Embaralhar o dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == "__main__":
    synthetic_df = generate_synthetic_transactions(num_transactions=2000, anomaly_ratio=0.05)
    # Salvar sem a coluna de ground truth para simular dados não rotulados para a aplicação
    synthetic_df_for_app = synthetic_df.drop(columns=['IsKnownAnomaly_GroundTruth'])
    
    output_filename = "synthetic_transactions.csv"
    synthetic_df_for_app.to_csv(output_filename, index=False)
    print(f"Dados sintéticos salvos em '{output_filename}'")

    # Para avaliação, você pode querer salvar com o ground truth também
    # synthetic_df.to_csv("synthetic_transactions_with_ground_truth.csv", index=False)
    # print(f"Dados sintéticos com ground truth salvos em 'synthetic_transactions_with_ground_truth.csv'")
