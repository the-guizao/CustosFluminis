import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

file_path = 'sentinela_rio_doce_dados.csv'
try:
    final_df_raw = pd.read_csv(file_path)
    print("Arquivo CSV local carregado com sucesso!")
except FileNotFoundError:
    print(f"ERRO: Arquivo não encontrado em '{file_path}'")
    exit()

#data
final_df = final_df_raw.copy()
final_df['date'] = pd.to_datetime(final_df['system:time_start'], unit='ms')
final_df = final_df[['date', 'ndvi', 'precipitacao_total_mes', 'temp_media_mes']]
final_df = final_df.set_index('date').dropna()
final_df_reset = final_df.reset_index()

print("\n--- Tabela de Dados Final (NDVI + Clima) ---")
print(final_df.head())
print("---------------------------------------------")


#ndvi
print("Gerando Gráfico 1: Tendência do NDVI...")
final_df_reset['days_since_start'] = (final_df_reset['date'] - final_df_reset['date'].min()).dt.days
X_ndvi = final_df_reset[['days_since_start']]; y_ndvi = final_df_reset['ndvi']
model_ndvi = LinearRegression(); model_ndvi.fit(X_ndvi, y_ndvi)
trend_ndvi = model_ndvi.predict(X_ndvi); slope_per_year_ndvi = model_ndvi.coef_[0] * 365
plt.figure(figsize=(12, 7)); plt.scatter(final_df_reset['date'], y_ndvi, alpha=0.6, label='NDVI Médio Mensal'); plt.plot(final_df_reset['date'], trend_ndvi, color='red', linewidth=3, label=f'Linha de Tendência (Anual: {slope_per_year_ndvi:+.4f})'); plt.title('Tendência do NDVI Mensal na APP do Rio Doce'); plt.xlabel('Data'); plt.ylabel('NDVI Médio'); plt.grid(True); plt.legend();
plt.savefig('grafico_1_tendencia_ndvi.png', dpi=300)
plt.show()


#linreg temp
print("Gerando Gráfico 2: Tendência da Temperatura...")
y_temp = final_df_reset['temp_media_mes']
model_temp = LinearRegression(); model_temp.fit(X_ndvi, y_temp) # Usa o mesmo X (tempo)
trend_temp = model_temp.predict(X_ndvi); slope_per_year_temp = model_temp.coef_[0] * 365
plt.figure(figsize=(12, 7)); plt.scatter(final_df_reset['date'], y_temp, alpha=0.6, label='Temp. Média Mensal', color='orange'); plt.plot(final_df_reset['date'], trend_temp, color='darkred', linewidth=3, label=f'Linha de Tendência (Anual: {slope_per_year_temp:+.4f} °C)'); plt.title('Tendência da Temperatura Média Mensal em Colatina'); plt.xlabel('Data'); plt.ylabel('Temperatura Média (°C)'); plt.grid(True); plt.legend();
plt.savefig('grafico_2_tendencia_temperatura.png', dpi=300)
plt.show()


#analise
print("Gerando Gráfico 3 e 4: Análise com XGBoost...")
df_xgb = final_df.reset_index().copy()
df_xgb['month'] = df_xgb['date'].dt.month
df_xgb['year'] = df_xgb['date'].dt.year
df_xgb['day_of_year'] = df_xgb['date'].dt.dayofyear
X_train = df_xgb[['year', 'month', 'day_of_year', 'precipitacao_total_mes', 'temp_media_mes']]
y_train = df_xgb['ndvi']
xgbr = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42)
xgbr.fit(X_train, y_train)

#variable imp
feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': xgbr.feature_importances_}).sort_values('importance', ascending=False)
plt.figure(figsize=(10, 6)); plt.barh(feature_importances['feature'], feature_importances['importance'], color='teal'); plt.xlabel('Importância'); plt.title('Importância de Cada Variável para o Modelo XGBoost'); plt.gca().invert_yaxis();
plt.savefig('grafico_3_importancia_variaveis.png', dpi=300)
plt.show()

#forecasting
last_date = df_xgb['date'].max()
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=36, freq='MS')
future_df = pd.DataFrame({'date': future_dates})
future_df['month'] = future_df['date'].dt.month
future_df['year'] = future_df['date'].dt.year
future_df['day_of_year'] = future_df['date'].dt.dayofyear
monthly_avg = df_xgb.groupby('month')[['precipitacao_total_mes', 'temp_media_mes']].mean()
future_df = pd.merge(future_df, monthly_avg, on='month', how='left')
X_future = future_df[X_train.columns]
future_prediction = xgbr.predict(X_future)
future_df['xgb_forecast'] = future_prediction
df_xgb['xgb_fit'] = xgbr.predict(X_train)

plt.figure(figsize=(15, 7))
plt.plot(df_xgb['date'], df_xgb['ndvi'], label='NDVI Observado', marker='o', linestyle='None', alpha=0.6)
plt.plot(df_xgb['date'], df_xgb['xgb_fit'], label='Ajuste XGBoost (Histórico)', color='purple', linestyle='-')
plt.plot(future_df['date'], future_df['xgb_forecast'], label='Previsão XGBoost (Futuro)', color='red', linestyle='--')
plt.title('Previsão de NDVI com XGBoost (Próximos 3 Anos)')
plt.xlabel('Data'); plt.ylabel('NDVI'); plt.axvline(x=last_date, color='r', linestyle=':', lw=2, label='Início da Predição'); plt.legend(); plt.grid(True);
plt.savefig('grafico_4_previsao_xgboost.png', dpi=300)
plt.show()

print("\nFim do Projeto! Todos os gráficos foram salvos como arquivos .png em alta resolução.")