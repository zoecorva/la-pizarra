"""Dashboard macroeconómico de Argentina: dólar, brecha, inflación y un modelo predictivo."""
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

BASE_URL = "https://api.argentinadatos.com/v1"

st.set_page_config(page_title="Macro Argentina", page_icon="📈", layout="wide")


# ---------- Carga de datos ----------
@st.cache_data(ttl=3600)
def get_json(path: str):
    r = requests.get(f"{BASE_URL}/{path}", timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600)
def load_dolar() -> pd.DataFrame:
    frames = []
    for casa in ["oficial", "blue"]:
        df = pd.DataFrame(get_json(f"cotizaciones/dolares/{casa}"))
        df["fecha"] = pd.to_datetime(df["fecha"])
        frames.append(df[["fecha", "casa", "venta"]])
    wide = (
        pd.concat(frames)
        .pivot_table(index="fecha", columns="casa", values="venta")
        .dropna()
        .sort_index()
    )
    wide["brecha_pct"] = (wide["blue"] / wide["oficial"] - 1) * 100
    return wide


@st.cache_data(ttl=3600)
def load_inflacion() -> pd.DataFrame:
    df = pd.DataFrame(get_json("finanzas/indices/inflacion"))
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").set_index("fecha").rename(columns={"valor": "mensual"})
    factor = 1 + df["mensual"] / 100
    df["interanual"] = (factor.rolling(12).apply(np.prod, raw=True) - 1) * 100
    return df


# ---------- Modelo ----------
N_LAGS = 6


def make_features(s: pd.Series) -> pd.DataFrame:
    X = pd.concat({f"lag_{i}": s.shift(i) for i in range(1, N_LAGS + 1)}, axis=1)
    X["media_3m"] = s.shift(1).rolling(3).mean()
    return X


@st.cache_data(ttl=3600)
def backtest_and_forecast(s: pd.Series, test_size: int = 12):
    """Walk-forward: reentrena mes a mes y compara contra un baseline ingenuo."""
    data = pd.concat([make_features(s), s.rename("y")], axis=1).dropna()
    feats = [c for c in data.columns if c != "y"]

    preds = []
    for i in range(len(data) - test_size, len(data)):
        train = data.iloc[:i]
        model = Ridge(alpha=1.0).fit(train[feats], train["y"])
        preds.append(model.predict(data.iloc[[i]][feats])[0])

    test = data.iloc[-test_size:]
    result = pd.DataFrame(
        {"real": test["y"], "modelo": preds, "ingenuo": test["lag_1"]}, index=test.index
    )
    metrics = {
        "MAE modelo": mean_absolute_error(result["real"], result["modelo"]),
        "MAE ingenuo": mean_absolute_error(result["real"], result["ingenuo"]),
    }

    # Pronóstico del próximo mes con el modelo entrenado sobre todo el historial
    final = Ridge(alpha=1.0).fit(data[feats], data["y"])
    last = {f"lag_{i}": s.iloc[-i] for i in range(1, N_LAGS + 1)}
    last["media_3m"] = s.iloc[-3:].mean()
    next_pred = final.predict(pd.DataFrame([last])[feats])[0]
    return result, metrics, next_pred


# ---------- UI ----------
st.title("📈 Macro Argentina")
st.caption("Dólar oficial vs. blue, inflación mensual (INDEC) y un modelo para el próximo dato. "
           "Fuente: argentinadatos.com")

try:
    dolar, infl = load_dolar(), load_inflacion()
except requests.RequestException as e:
    st.error(f"No pude descargar los datos: {e}")
    st.stop()

with st.sidebar:
    st.header("Filtros")
    rango = st.date_input(
        "Rango de fechas (dólar)",
        value=(dolar.index.min().date(), dolar.index.max().date()),
        min_value=dolar.index.min().date(),
        max_value=dolar.index.max().date(),
    )

if isinstance(rango, tuple) and len(rango) == 2:
    d = dolar.loc[str(rango[0]):str(rango[1])]
else:
    d = dolar

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dólar oficial", f"${dolar['oficial'].iloc[-1]:,.0f}")
c2.metric("Dólar blue", f"${dolar['blue'].iloc[-1]:,.0f}")
c3.metric("Brecha", f"{dolar['brecha_pct'].iloc[-1]:.1f}%")
c4.metric("Inflación último mes", f"{infl['mensual'].iloc[-1]:.1f}%")

tab1, tab2, tab3 = st.tabs(["💵 Dólar", "🛒 Inflación", "🤖 Modelo"])

with tab1:
    fig = px.line(d, y=["oficial", "blue"], labels={"value": "ARS", "fecha": "", "variable": ""})
    st.plotly_chart(fig, use_container_width=True)
    fig = px.area(d, y="brecha_pct", labels={"brecha_pct": "Brecha (%)", "fecha": ""})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.bar(infl.tail(60), y="mensual", labels={"mensual": "Inflación mensual (%)", "fecha": ""})
    st.plotly_chart(fig, use_container_width=True)
    fig = px.line(infl.dropna().tail(120), y="interanual",
                  labels={"interanual": "Inflación interanual (%)", "fecha": ""})
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Predicción de la inflación mensual")
    st.write("Regresión Ridge con 6 rezagos + media móvil de 3 meses, validada con "
             "*walk-forward* sobre los últimos 12 meses y comparada con un baseline ingenuo "
             "(repetir el mes anterior).")
    result, metrics, next_pred = backtest_and_forecast(infl["mensual"])
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE modelo (p.p.)", f"{metrics['MAE modelo']:.2f}")
    m2.metric("MAE ingenuo (p.p.)", f"{metrics['MAE ingenuo']:.2f}")
    m3.metric("Próximo mes (estimado)", f"{next_pred:.1f}%")
    fig = px.line(result, y=["real", "modelo", "ingenuo"],
                  labels={"value": "Inflación mensual (%)", "fecha": "", "variable": ""})
    st.plotly_chart(fig, use_container_width=True)
    st.info("Con series cortas y cambios de régimen, el baseline ingenuo suele ser difícil de "
            "superar. Reportarlo con honestidad es parte del análisis.")
