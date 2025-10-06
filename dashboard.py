import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Custos Fluminis")

st.title("🛰️ Custos Fluminis: Dashboard Interativo")
st.markdown("Uma ferramenta para simulação de cenários climáticos futuros para a saúde da vegetação (NDVI) da APP de Colatina, ES.")

@st.cache_data
def load_data(file_path):
    df_raw = pd.read_csv(file_path)
    df = df_raw.copy()
    df['date'] = pd.to_datetime(df['system:time_start'], unit='ms')
    df = df[['date', 'ndvi', 'precipitacao_total_mes', 'temp_media_mes', 'co_media_mensal', 'aerosol_media_mensal']]
    df = df.set_index('date').dropna()
    return df

@st.cache_data
def train_model(df):
    df_model = df.reset_index().copy()
    df_model['month'] = df_model['date'].dt.month
    df_model['year'] = df_model['date'].dt.year

    X = df_model[['year', 'month', 'precipitacao_total_mes', 'temp_media_mes', 'co_media_mensal', 'aerosol_media_mensal']]
    y = df_model['ndvi']

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X, y)
    
    df_model['xgb_fit'] = model.predict(X)
    return model, df_model

try:
    data = load_data('sentinela_dados_finais.csv')
    model, df_model_fit = train_model(data)
    st.success("Dados carregados e modelo XGBoost treinado com sucesso!")
except FileNotFoundError:
    st.error("ERRO: Ficheiro 'sentinela_dados_finais.csv' não encontrado. Por favor, coloque o ficheiro na mesma pasta que este script.")
    st.stop()

st.sidebar.header("Painel de Simulação de Cenários")
st.sidebar.markdown("Ajuste os modificadores para testar o impacto de diferentes futuros climáticos.")

future_steps = st.sidebar.slider(
    'Horizonte de Previsão (meses)', 
    min_value=12, max_value=60, value=36, step=12
)

precip_modifier = st.sidebar.slider(
    'Modificador de Precipitação (%)', 
    min_value=-50, max_value=50, value=-30, step=5,
    format='%d%%'
) / 100.0 + 1.0

temp_modifier = st.sidebar.slider(
    'Modificador de Temperatura (°C)', 
    min_value=-2.0, max_value=5.0, value=1.5, step=0.5,
    format='%.1f°C'
)

def run_simulation(model, df_historical, precip_mod, temp_add):
    last_date = df_historical['date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_steps, freq='MS')
    future_df = pd.DataFrame({'date': future_dates})
    
    future_df['month'] = future_df['date'].dt.month
    future_df['year'] = future_df['date'].dt.year
    
    monthly_avg = df_historical.groupby('month')[['precipitacao_total_mes', 'temp_media_mes', 'co_media_mensal', 'aerosol_media_mensal']].mean()
    
    # Aplica os modificadores do cenário
    future_df['precipitacao_total_mes'] = future_df['month'].map(lambda m: monthly_avg.loc[m, 'precipitacao_total_mes'] * precip_mod)
    future_df['temp_media_mes'] = future_df['month'].map(lambda m: monthly_avg.loc[m, 'temp_media_mes'] + temp_add) # Modificador aditivo para temperatura
    future_df['co_media_mensal'] = future_df['month'].map(lambda m: monthly_avg.loc[m, 'co_media_mensal'])
    future_df['aerosol_media_mensal'] = future_df['month'].map(lambda m: monthly_avg.loc[m, 'aerosol_media_mensal'])
    
    X_future = future_df[df_historical[['year', 'month', 'precipitacao_total_mes', 'temp_media_mes', 'co_media_mensal', 'aerosol_media_mensal']].columns]
    future_prediction = model.predict(X_future)
    future_df['forecast'] = future_prediction
    
    return future_df[['date', 'forecast']].set_index('date')

st.header("Visualização da Simulação")

forecast_normal = run_simulation(model, df_model_fit, 1.0, 0.0)
scenario_name = f'Cenário: {precip_modifier*100-100:+.0f}% Chuva, {temp_modifier:+.1f}°C Temp.'
forecast_cenario = run_simulation(model, df_model_fit, precip_modifier, temp_modifier)

fig, ax = plt.subplots(figsize=(18, 9))
ax.plot(df_model_fit['date'], df_model_fit['ndvi'], label='NDVI Observado', marker='o', linestyle='None', alpha=0.5, color='gray')
ax.plot(df_model_fit['date'], df_model_fit['xgb_fit'], label='Ajuste XGBoost (Histórico)', color='purple', linestyle='-')
ax.plot(forecast_normal.index, forecast_normal, label='Previsão (Cenário Normal)', linestyle='--', color='blue')
ax.plot(forecast_cenario.index, forecast_cenario, label=scenario_name, linestyle='--', color='red')
ax.set_title('Simulação de Cenários Climáticos Futuros para o NDVI com XGBoost', fontsize=16)
ax.set_xlabel('Data'); ax.set_ylabel('NDVI'); ax.axvline(x=df_model_fit['date'].max(), color='black', linestyle=':', lw=2, label='Início da Simulação');
ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
ax.set_ylim(bottom=0)

st.pyplot(fig)

st.markdown("---")
st.markdown("Desenvolvido no âmbito do Projeto Sentinela do Rio Doce - Marista Colatina")
