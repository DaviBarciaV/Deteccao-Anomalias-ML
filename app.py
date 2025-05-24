import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go

# Configuração da página do Streamlit
st.set_page_config(layout="wide", page_title="Detecção de Anomalias Financeiras")

# Título da Aplicação
st.title("📊 Sistema de Detecção de Anomalias em Transações Financeiras")
st.markdown("""
Bem-vindo! Esta aplicação permite identificar transações financeiras potencialmente anômalas ou fraudulentas
usando algoritmos de Machine Learning não supervisionados.
""")

# --- Funções Auxiliares ---
def load_data(uploaded_file):
    """Carrega os dados do arquivo CSV."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {e}")
            return None
    return None

def preprocess_data(df, selected_features, missing_value_strategy='median'):
    """Pré-processa os dados: seleciona features, trata valores ausentes e codifica categóricas."""
    df_processed = df[selected_features].copy()

    # Identificar colunas numéricas e categóricas
    numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_processed.select_dtypes(include='object').columns.tolist()

    # Tratar valores ausentes em colunas numéricas
    if numerical_cols:
        if missing_value_strategy == 'median':
            imputer_numerical = SimpleImputer(strategy='median')
        elif missing_value_strategy == 'mean':
            imputer_numerical = SimpleImputer(strategy='mean')
        else: # drop
            df_processed.dropna(subset=numerical_cols, inplace=True) # Simplificado para o exemplo
            imputer_numerical = None # Não será usado

        if imputer_numerical and not df_processed[numerical_cols].empty:
             # Verificar se há dados após o drop, se for o caso
            if not df_processed[numerical_cols].isnull().all().all(): # Evitar imputar se tudo for NaN
                df_processed[numerical_cols] = imputer_numerical.fit_transform(df_processed[numerical_cols])


    # Tratar valores ausentes e codificar colunas categóricas
    for col in categorical_cols:
        # Imputar com a moda
        if df_processed[col].isnull().any():
            mode = df_processed[col].mode()[0] if not df_processed[col].mode().empty else "Unknown"
            df_processed[col].fillna(mode, inplace=True)
        # Label Encoding
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])

    # Escalar features numéricas (após a codificação das categóricas, pois elas se tornam numéricas)
    # Re-identificar colunas numéricas após LabelEncoding
    all_numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    if all_numerical_cols:
        scaler = StandardScaler()
        df_processed[all_numerical_cols] = scaler.fit_transform(df_processed[all_numerical_cols])

    return df_processed


# --- Sidebar para Controles ---
st.sidebar.header("⚙️ Configurações")
uploaded_file = st.sidebar.file_uploader("Carregue seu arquivo CSV de transações", type=["csv"])

df_original = None
df_processed = None
selected_features_for_model = []

if uploaded_file:
    df_original = load_data(uploaded_file)
    if df_original is not None:
        st.sidebar.success("Arquivo carregado com sucesso!")

        st.subheader("Visualização dos Dados Originais (Primeiras 5 Linhas)")
        st.dataframe(df_original.head())

        st.sidebar.markdown("---")
        st.sidebar.subheader("1. Seleção de Features")
        st.sidebar.markdown("Escolha as colunas para análise (numéricas e categóricas relevantes).")

        # Tenta identificar colunas numéricas e categóricas automaticamente
        default_numerical_cols = df_original.select_dtypes(include=np.number).columns.tolist()
        default_categorical_cols = df_original.select_dtypes(include='object').columns.tolist()
        
        all_columns = df_original.columns.tolist()
        selected_features_for_model = st.sidebar.multiselect(
            "Selecione as features para o modelo:",
            options=all_columns,
            default=default_numerical_cols + default_categorical_cols # Sugere todas as colunas por padrão
        )

        if not selected_features_for_model:
            st.sidebar.warning("Por favor, selecione ao menos uma feature.")
        else:
            st.sidebar.markdown("---")
            st.sidebar.subheader("2. Pré-processamento")
            missing_value_strategy = st.sidebar.selectbox(
                "Estratégia para valores ausentes (numéricos):",
                ('median', 'mean', 'drop_rows_with_missing_values') # 'drop' é simplificado aqui
            )

            st.sidebar.markdown("---")
            st.sidebar.subheader("3. Algoritmo de Detecção")
            algorithm = st.sidebar.selectbox(
                "Escolha o algoritmo:",
                ("Isolation Forest", "One-Class SVM")
            )

            # Parâmetros do modelo
            st.sidebar.subheader("Parâmetros do Modelo")
            if algorithm == "Isolation Forest":
                contamination_if = st.sidebar.slider(
                    "Contaminação (proporção esperada de outliers):",
                    min_value=0.01, max_value=0.5, value=0.05, step=0.01,
                    help="A proporção de outliers no conjunto de dados."
                )
                n_estimators_if = st.sidebar.slider(
                    "Número de Estimadores (Árvores):",
                    min_value=50, max_value=500, value=100, step=10
                )
            elif algorithm == "One-Class SVM":
                nu_svm = st.sidebar.slider(
                    "Nu (limite superior na fração de outliers e inferior na fração de vetores de suporte):",
                    min_value=0.01, max_value=0.5, value=0.05, step=0.01
                )
                kernel_svm = st.sidebar.selectbox(
                    "Kernel:",
                    ('rbf', 'linear', 'poly', 'sigmoid'),
                    index=0
                )
                gamma_svm = st.sidebar.select_slider(
                    "Gamma (coeficiente do kernel para 'rbf', 'poly', 'sigmoid'):",
                    options=['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    value='scale'
                )

            if st.sidebar.button("🚀 Detectar Anomalias", use_container_width=True):
                if not selected_features_for_model:
                    st.error("Nenhuma feature selecionada para o modelo.")
                else:
                    with st.spinner("Pré-processando dados e treinando o modelo..."):
                        # Pré-processamento
                        df_for_model = preprocess_data(df_original, selected_features_for_model, missing_value_strategy)

                        if df_for_model.empty or df_for_model.isnull().all().all():
                            st.error("Os dados ficaram vazios ou todos NaN após o pré-processamento. Verifique as features selecionadas e a estratégia de valores ausentes.")
                        else:
                            st.subheader("Dados Pré-processados (Normalizados/Codificados)")
                            st.dataframe(df_for_model.head())

                            # Treinamento do Modelo
                            if algorithm == "Isolation Forest":
                                model = IsolationForest(n_estimators=n_estimators_if, contamination=contamination_if, random_state=42)
                            elif algorithm == "One-Class SVM":
                                model = OneClassSVM(nu=nu_svm, kernel=kernel_svm, gamma=gamma_svm)

                            model.fit(df_for_model)
                            predictions = model.predict(df_for_model)

                            # Adicionar predições ao DataFrame original para análise
                            # -1 para anomalias, 1 para normais. Mapear para 1 (anomalia) e 0 (normal)
                            df_original['anomaly_score'] = model.decision_function(df_for_model) if hasattr(model, 'decision_function') else np.nan
                            df_original['is_anomaly'] = np.where(predictions == -1, 1, 0)

                            st.success(f"Detecção de anomalias concluída usando {algorithm}!")

                            # --- Resultados ---
                            st.markdown("---")
                            st.subheader("Resultados da Detecção de Anomalias")

                            # Contagem de anomalias
                            anomaly_counts = df_original['is_anomaly'].value_counts()
                            st.metric(label="Transações Normais", value=anomaly_counts.get(0, 0))
                            st.metric(label="Transações Anômalas", value=anomaly_counts.get(1, 0))

                            # Exibir transações anômalas
                            st.subheader("Transações Identificadas como Anômalas")
                            anomalous_transactions = df_original[df_original['is_anomaly'] == 1]
                            if anomalous_transactions.empty:
                                st.info("Nenhuma anomalia detectada com os parâmetros atuais.")
                            else:
                                st.dataframe(anomalous_transactions)

                            # --- Visualizações ---
                            st.markdown("---")
                            st.subheader("Visualizações Interativas")

                            # Seleção de features para visualização
                            numerical_cols_original = df_original.select_dtypes(include=np.number).columns.tolist()
                            # Remover 'is_anomaly' e 'anomaly_score' da seleção padrão para eixos, se existirem
                            features_for_plot = [col for col in numerical_cols_original if col not in ['is_anomaly', 'anomaly_score']]


                            if len(features_for_plot) >= 2:
                                col1, col2 = st.columns(2)
                                with col1:
                                    x_axis = st.selectbox("Selecione a feature para o Eixo X:", features_for_plot, index=0)
                                with col2:
                                    y_axis_options = [f for f in features_for_plot if f != x_axis]
                                    if y_axis_options:
                                        y_axis = st.selectbox("Selecione a feature para o Eixo Y:", y_axis_options, index=0 if len(y_axis_options) > 0 else None)
                                    else:
                                        y_axis = x_axis # Fallback se só houver uma feature numérica

                                if x_axis and y_axis:
                                    fig_scatter = px.scatter(
                                        df_original,
                                        x=x_axis,
                                        y=y_axis,
                                        color='is_anomaly',
                                        color_discrete_map={0: 'blue', 1: 'red'},
                                        title=f"Scatter Plot: {x_axis} vs {y_axis} (Anomalias em Vermelho)",
                                        hover_data=df_original.columns
                                    )
                                    st.plotly_chart(fig_scatter, use_container_width=True)
                            elif len(features_for_plot) == 1:
                                x_axis = features_for_plot[0]
                                fig_hist = go.Figure()
                                fig_hist.add_trace(go.Histogram(x=df_original[df_original['is_anomaly']==0][x_axis], name='Normal', marker_color='blue', opacity=0.7))
                                fig_hist.add_trace(go.Histogram(x=df_original[df_original['is_anomaly']==1][x_axis], name='Anomalia', marker_color='red', opacity=0.7))
                                fig_hist.update_layout(barmode='overlay', title=f"Distribuição de {x_axis} por Status de Anomalia")
                                st.plotly_chart(fig_hist, use_container_width=True)
                            else:
                                st.warning("Selecione ao menos uma feature numérica para visualização.")


                            # Distribuição dos Scores de Anomalia (se disponível)
                            if 'anomaly_score' in df_original.columns and not df_original['anomaly_score'].isnull().all():
                                fig_score_dist = px.histogram(
                                    df_original,
                                    x='anomaly_score',
                                    color='is_anomaly',
                                    marginal="box", # Adiciona box plot nas margens
                                    title="Distribuição dos Scores de Anomalia",
                                    color_discrete_map={0: 'blue', 1: 'red'}
                                )
                                st.plotly_chart(fig_score_dist, use_container_width=True)

else:
    st.info("ℹ️ Por favor, carregue um arquivo CSV para começar.")