import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

file_path = 'sentinela_dados_finais.csv' 
try:
    final_df_raw = pd.read_csv(file_path)
    print(f"Arquivo CSV '{file_path}' carregado com sucesso!")
except FileNotFoundError:
    print(f"ERRO: Arquivo não encontrado em '{file_path}'")
    exit()

final_df = final_df_raw.copy()
final_df['date'] = pd.to_datetime(final_df['system:time_start'], unit='ms')
final_df = final_df[['date', 'ndvi', 'precipitacao_total_mes', 'temp_media_mes', 'co_media_mensal', 'aerosol_media_mensal']]
final_df = final_df.set_index('date').dropna()
print(f"\n{len(final_df)} meses com dados completos para análise.")

# --- PASSO 2: PREPARAÇÃO E TREINAMENTO DO MODELO XGBOOST ---
print("\nIniciando preparação e treinamento do modelo XGBoost...")

df_model = final_df.reset_index().copy()
df_model['month'] = df_model['date'].dt.month
df_model['year'] = df_model['date'].dt.year

X = df_model[['year', 'month', 'precipitacao_total_mes', 'temp_media_mes', 'co_media_mensal', 'aerosol_media_mensal']]
y = df_model['ndvi']

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X, y)
print("Modelo XGBoost treinado com sucesso!")

# Adiciona a previsão sobre os dados históricos (ajuste) ao dataframe
df_model['xgb_fit'] = model.predict(X)

# --- PASSO 3: IA EXPLICÁVEL (XAI) COM SHAP ---
print("\nIniciando análise de interpretabilidade com SHAP...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("Gerando gráfico de sumário SHAP (Beeswarm)...")
plt.title('Impacto de Cada Variável na Previsão do NDVI (SHAP)')
shap.summary_plot(shap_values, X, show=False)
plt.savefig('grafico_shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
plt.show()

print("Gerando gráfico de importância das variáveis (SHAP)...")
plt.title('Importância Média das Variáveis (SHAP)')
shap.summary_plot(shap_values, X, plot_type='bar', show=False)
plt.savefig('grafico_shap_summary_bar.png', dpi=300, bbox_inches='tight')
plt.show()


# --- PASSO 4: GÊMEO DIGITAL - SIMULAÇÃO DE CENÁRIOS ---
print("\nIniciando simulação de cenários climáticos...")

future_steps = 36
precip_modifier = 0.7 
temp_modifier = 1.05  

def run_simulation(scenario_name, precip_mod, temp_mod):
    last_date = df_model['date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_steps, freq='MS')
    future_df = pd.DataFrame({'date': future_dates})
    
    future_df['month'] = future_df['date'].dt.month
    future_df['year'] = future_df['date'].dt.year
    
    monthly_avg = df_model.groupby('month')[['precipitacao_total_mes', 'temp_media_mes', 'co_media_mensal', 'aerosol_media_mensal']].mean()
    
    future_df['precipitacao_total_mes'] = future_df['month'].map(lambda m: monthly_avg.loc[m, 'precipitacao_total_mes'] * precip_mod)
    future_df['temp_media_mes'] = future_df['month'].map(lambda m: monthly_avg.loc[m, 'temp_media_mes'] * temp_mod)
    future_df['co_media_mensal'] = future_df['month'].map(lambda m: monthly_avg.loc[m, 'co_media_mensal'])
    future_df['aerosol_media_mensal'] = future_df['month'].map(lambda m: monthly_avg.loc[m, 'aerosol_media_mensal'])
    
    X_future = future_df[X.columns]
    future_prediction = model.predict(X_future)
    future_df[scenario_name] = future_prediction
    
    return future_df[['date', scenario_name]].set_index('date')

forecast_normal = run_simulation('Cenário Normal', 1.0, 1.0)
scenario_name = f'Cenário: {precip_modifier*100-100:+.0f}% Chuva, {temp_modifier*100-100:+.0f}% Temp.'
forecast_cenario = run_simulation(scenario_name, precip_modifier, temp_modifier)

# Visualização da Simulação
print("Gerando gráfico de comparação de cenários...")
plt.figure(figsize=(18, 9))
plt.plot(df_model['date'], df_model['ndvi'], label='NDVI Observado', marker='o', linestyle='None', alpha=0.5)
# ==============================================================================
# >>>>> LINHA ADICIONADA AQUI <<<<<
plt.plot(df_model['date'], df_model['xgb_fit'], label='Ajuste XGBoost (Histórico)', color='purple', linestyle='-')
# ==============================================================================
plt.plot(forecast_normal.index, forecast_normal, label='Previsão (Cenário Normal)', linestyle='--', color='blue')
plt.plot(forecast_cenario.index, forecast_cenario, label=scenario_name, linestyle='--', color='red')
plt.title('Gêmeo Digital: Simulação de Cenários Climáticos Futuros para o NDVI com XGBoost', fontsize=16)
plt.xlabel('Data'); plt.ylabel('NDVI'); plt.axvline(x=df_model['date'].max(), color='black', linestyle=':', lw=2, label='Início da Simulação');
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('grafico_simulacao_cenarios_xgboost.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFim do Projeto!")