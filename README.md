# 📈 Macro Argentina — Dashboard con Streamlit

Dashboard interactivo que descarga datos públicos de la economía argentina y muestra:

- **Dólar oficial vs. blue** y la evolución de la **brecha cambiaria**.
- **Inflación mensual e interanual** (INDEC).
- **Modelo predictivo** (Ridge con rezagos) para la inflación del próximo mes, con
  validación *walk-forward* y comparación contra un baseline ingenuo.

## Stack
Python · pandas · scikit-learn · Plotly · Streamlit · API de [argentinadatos.com](https://argentinadatos.com)

## Cómo correrlo
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Decisiones de diseño
- `st.cache_data(ttl=3600)` para no golpear la API en cada interacción.
- Validación temporal (sin mezclar pasado y futuro) en lugar de `train_test_split` aleatorio.
- Se reporta el MAE del modelo **y** del baseline ingenuo.

## Próximos pasos
- Sumar riesgo país, plazo fijo y otras variables como features.
- Probar otros modelos (Gradient Boosting, SARIMA, Prophet).
- Deploy en Streamlit Community Cloud.
