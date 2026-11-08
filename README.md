# La Pizarra

**Dólar, inflación y tasas de Argentina en un solo tablero.** Dashboard interactivo hecho con Streamlit.

Dashboard interactivo que descarga datos públicos de la economía argentina y muestra:

- **Dólar oficial, blue, MEP y CCL**, con variación a 30 días y **brechas** contra el oficial.
- **Filtros de período** (1 mes a "Todo", o fechas personalizadas) y selector de cotizaciones.
- **Inflación mensual e interanual** (INDEC) con variación contra el mes anterior.
- **Datos oficiales del BCRA** (API v4.0): explorador de las principales variables (reservas, tasas, base monetaria, etc.).
- **Plazo fijo por banco**: ranking de tasas de todos los bancos, interés ganado y rendimiento real contra la inflación.
- **Calculadora de inflación**: cuánto equivale hoy un monto de cualquier mes anterior.
- **Modelo predictivo** (Ridge con rezagos y medias móviles) para la inflación del próximo mes,
  con validación *walk-forward*, comparación contra un baseline ingenuo y un ensemble 50/50.

Diseño responsive: se adapta a celulares (indicadores en dos columnas, filtros en un desplegable).

## Stack
Python · pandas · scikit-learn · Plotly · Streamlit · APIs de [argentinadatos.com](https://argentinadatos.com) y del [BCRA](https://www.bcra.gob.ar/apis-banco-central/)

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
