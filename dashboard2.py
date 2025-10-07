# ==============================================================================
# PROJETO SENTINELA DO RIO DOCE - DASHBOARD INTERATIVO v2.1
# ==============================================================================
# Versão refatorada para maior robustez do código de simulação,
# corrigindo o erro de 'KeyError' e melhorando a legibilidade.
#
# Para executar:
# 1. Instale: pip install streamlit pandas xgboost scikit-learn tensorflow plotly
# 2. Execute: streamlit run dashboard.py
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings('ignore')

#pagelayout
st.set_page_config(layout="wide", page_title="Custos Fluminis")

st.title("🛰️ Custos Fluminis: Dashboard Interativo")
st.markdown("Uma ferramenta para simulação de cenários climáticos futuros para a saúde da vegetação (NDVI) da APP de Colatina, ES.")

#cache
@st.cache_data
def load_data(file_path):
    df_raw = pd.read_csv(file_path)
    df = df_raw.copy()
    df['date'] = pd.to_datetime(df['system:time_start'], unit='ms')
    df = df[['date', 'ndvi', 'precipitacao_total_mes', 'temp_media_mes', 'co_media_mensal', 'aerosol_media_mensal']]
    df = df.set_index('date').dropna()
    return df

@st.cache_resource
def train_xgb_model(df):
    st.info("Treinando modelo XGBoost... (executado apenas uma vez)")
    df_model = df.reset_index().copy()
    df_model['month'] = df_model['date'].dt.month
    df_model['year'] = df_model['date'].dt.year
    
    #features
    features = ['year', 'month', 'precipitacao_total_mes', 'temp_media_mes', 'co_media_mensal', 'aerosol_media_mensal']
    X = df_model[features]
    y = df_model['ndvi']
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X, y)
    df_model['fit'] = model.predict(X)
    return model, df_model, features

@st.cache_resource
def train_lstm_model(df):
    st.info("Treinando modelo LSTM")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    scaler_ndvi = MinMaxScaler(feature_range=(0, 1))
    scaler_ndvi.fit(df[['ndvi']])
    
    look_back = 6
    trainX, trainY = [], []
    for i in range(len(scaled_data) - look_back):
        trainX.append(scaled_data[i:(i + look_back), :])
        trainY.append(scaled_data[i + look_back, 0])
    trainX, trainY = np.array(trainX), np.array(trainY)

    model = Sequential()
    model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=150, batch_size=1, verbose=0)
    
    fit_scaled = model.predict(trainX)
    fit_descaled = scaler_ndvi.inverse_transform(fit_scaled)
    
    df_model = df.reset_index()
    df_model.loc[look_back:, 'fit'] = fit_descaled.flatten()
    
    return model, df_model, scaler, scaler_ndvi, look_back

#data
try:
    data = load_data('sentinela_dados_finais.csv')
except FileNotFoundError:
    st.error("ERRO: Ficheiro 'sentinela_dados_finais.csv' não encontrado.")
    st.stop()

#data
st.sidebar.header("Painel de Controlo")
model_choice = st.sidebar.selectbox(
    'Escolha o Modelo de Previsão',
    ('XGBoost', 'LSTM')
)

st.sidebar.header("Simulação de Cenários")
future_steps = st.sidebar.slider('Horizonte de Previsão (meses)', 12, 60, 36, 12)
precip_modifier = st.sidebar.slider('Modificador de Precipitação (%)', -50, 50, -30, 5, format='%d%%') / 100.0 + 1.0
temp_modifier = st.sidebar.slider('Modificador de Temperatura (°C)', -2.0, 5.0, 1.5, 0.5, format='%.1f°C')

#training
st.header(f"Visualização da Simulação com {model_choice}")

# Bloco de lógica para o XGBoost
if model_choice == 'XGBoost':
    xgb_model, xgb_model_fit, xgb_features = train_xgb_model(data)
    df_to_plot = xgb_model_fit
    
    # Lógica de simulação para XGBoost
    last_date = df_to_plot['date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_steps, freq='MS')
    future_df = pd.DataFrame({'date': future_dates})
    future_df['month'] = future_df['date'].dt.month
    future_df['year'] = future_df['date'].dt.year
    
    monthly_avg = df_to_plot.groupby('month')[['precipitacao_total_mes', 'temp_media_mes', 'co_media_mensal', 'aerosol_media_mensal']].mean()
    
    # Função para criar o dataframe futuro
    def create_future_df(precip_mod, temp_add):
        df = future_df.copy()
        df['precipitacao_total_mes'] = df['month'].map(lambda m: monthly_avg.loc[m, 'precipitacao_total_mes'] * precip_mod)
        df['temp_media_mes'] = df['month'].map(lambda m: monthly_avg.loc[m, 'temp_media_mes'] + temp_add)
        df['co_media_mensal'] = df['month'].map(lambda m: monthly_avg.loc[m, 'co_media_mensal'])
        df['aerosol_media_mensal'] = df['month'].map(lambda m: monthly_avg.loc[m, 'aerosol_media_mensal'])
        X_future = df[xgb_features]
        df['forecast'] = xgb_model.predict(X_future)
        return df.set_index('date')

    forecast_normal = create_future_df(1.0, 0.0)
    forecast_cenario = create_future_df(precip_modifier, temp_modifier)

# Bloco de lógica para o LSTM
else: # LSTM
    lstm_model, lstm_model_fit, lstm_scaler, lstm_scaler_ndvi, lstm_look_back = train_lstm_model(data)
    df_to_plot = lstm_model_fit

    # Lógica de simulação para LSTM
    def create_lstm_forecast(precip_mod, temp_add):
        scaled_data = lstm_scaler.transform(data)
        last_sequence = scaled_data[-lstm_look_back:]
        current_sequence = last_sequence.reshape(1, lstm_look_back, scaled_data.shape[1])
        future_predictions_scaled = []
        
        future_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=future_steps, freq='MS')
        monthly_avg = df_to_plot.groupby(df_to_plot['date'].dt.month)[['precipitacao_total_mes', 'temp_media_mes', 'co_media_mensal', 'aerosol_media_mensal']].mean()

        for date in future_dates:
            next_pred_scaled = lstm_model.predict(current_sequence, verbose=0)
            future_predictions_scaled.append(next_pred_scaled[0, 0])
            
            month = date.month
            future_precip = monthly_avg.loc[month, 'precipitacao_total_mes'] * precip_mod
            future_temp = monthly_avg.loc[month, 'temp_media_mes'] + temp_add
            future_co = monthly_avg.loc[month, 'co_media_mensal']
            future_aerosol = monthly_avg.loc[month, 'aerosol_media_mensal']

            temp_step = [0, future_precip, future_temp, future_co, future_aerosol]
            scaled_exog = lstm_scaler.transform([temp_step])[0]
            new_step = np.insert(np.delete(scaled_exog, 0), 0, next_pred_scaled[0,0])
            current_sequence = np.append(current_sequence[:, 1:, :], [[new_step]], axis=1)
            
        future_prediction = lstm_scaler_ndvi.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1)).flatten()
        return pd.DataFrame({'forecast': future_prediction}, index=future_dates)

    forecast_normal = create_lstm_forecast(1.0, 0.0)
    forecast_cenario = create_lstm_forecast(precip_modifier, temp_modifier)


#grafico
fig = go.Figure()

scenario_name = f'Cenário: {precip_modifier*100-100:+.0f}% Chuva, {temp_modifier:+.1f}°C Temp.'

fig.add_trace(go.Scatter(x=df_to_plot['date'], y=df_to_plot['ndvi'], mode='markers', name='NDVI Observado', marker=dict(color='gray', opacity=0.6)))
fig.add_trace(go.Scatter(x=df_to_plot['date'], y=df_to_plot['fit'], mode='lines', name=f'Ajuste {model_choice} (Histórico)', line=dict(color='purple')))
fig.add_trace(go.Scatter(x=forecast_normal.index, y=forecast_normal['forecast'], mode='lines', name='Previsão (Cenário Normal)', line=dict(color='blue', dash='dash')))
fig.add_trace(go.Scatter(x=forecast_cenario.index, y=forecast_cenario['forecast'], mode='lines', name=scenario_name, line=dict(color='red', dash='dash')))
fig.add_vline(x=df_to_plot['date'].max(), line_width=2, line_dash="dash", line_color="black")

fig.update_layout(
    title=f'Simulação de Cenários Climáticos Futuros com {model_choice}',
    xaxis_title='Data', yaxis_title='NDVI', legend=dict(
        x=0.5,  # Center the legend horizontally
        y=-0.2, # Position the legend below the plot area
        xanchor='center', # Anchor the legend's x-coordinate to its center
        yanchor='top', # Anchor the legend's y-coordinate to its top
        orientation='h' # Arrange legend items horizontally
    ), legend_title='Legenda',
    height=600, template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("Desenvolvido no âmbito do Projeto Sentinela do Rio Doce - Marista Colatina")
st.markdown("Desenvolvido pelo Professor Me. Guilherme Schultz Netto")
st.image("grafico_3_editado.png")



