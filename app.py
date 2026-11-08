"""Dashboard macroeconómico de Argentina: dólares, brechas, inflación, calculadora y modelo."""
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

BASE_URL = "https://api.argentinadatos.com/v1"
CASAS = {"oficial": "Oficial", "blue": "Blue", "bolsa": "MEP", "contadoconliqui": "CCL"}
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
PERIODOS = {"1 mes": 30, "3 meses": 90, "6 meses": 180, "1 año": 365, "3 años": 1095, "Todo": None}

st.set_page_config(page_title="Macro Argentina", page_icon="📈", layout="wide")


def ars(x: float) -> str:
    return f"${x:,.0f}".replace(",", ".")


# ---------- Datos ----------
@st.cache_data(ttl=3600)
def get_json(path: str):
    r = requests.get(f"{BASE_URL}/{path}", timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600)
def load_dolar() -> pd.DataFrame:
    series = {}
    for casa, nombre in CASAS.items():
        try:
            df = pd.DataFrame(get_json(f"cotizaciones/dolares/{casa}"))
        except requests.RequestException:
            continue
        df["fecha"] = pd.to_datetime(df["fecha"])
        series[nombre] = df.drop_duplicates("fecha").set_index("fecha")["venta"]
    if "Oficial" not in series:
        raise requests.RequestException("No se pudo cargar el dólar oficial")
    wide = pd.DataFrame(series).sort_index().ffill(limit=5)
    return wide.dropna(subset=["Oficial"])


@st.cache_data(ttl=3600)
def load_inflacion() -> pd.DataFrame:
    df = pd.DataFrame(get_json("finanzas/indices/inflacion"))
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").set_index("fecha").rename(columns={"valor": "mensual"})
    factor = 1 + df["mensual"] / 100
    df["interanual"] = (factor.rolling(12).apply(np.prod, raw=True) - 1) * 100
    return df


def change(series: pd.Series, days: int = 30, pct: bool = True):
    """Variación entre el último dato y el de hace `days` días (en % o en puntos)."""
    s = series.dropna()
    ref = s[s.index <= s.index[-1] - pd.Timedelta(days=days)]
    if ref.empty:
        return None
    return (s.iloc[-1] / ref.iloc[-1] - 1) * 100 if pct else s.iloc[-1] - ref.iloc[-1]


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
    final = Ridge(alpha=1.0).fit(data[feats], data["y"])
    last = {f"lag_{i}": s.iloc[-i] for i in range(1, N_LAGS + 1)}
    last["media_3m"] = s.iloc[-3:].mean()
    return result, metrics, final.predict(pd.DataFrame([last])[feats])[0]


# ---------- App ----------
st.title("📈 Macro Argentina")

try:
    dolar, infl = load_dolar(), load_inflacion()
except requests.RequestException as e:
    st.error(f"No pude descargar los datos: {e}")
    st.stop()

# Sidebar: período y cotizaciones
with st.sidebar:
    st.header("Filtros")
    periodo = st.radio("Período", list(PERIODOS) + ["Personalizado"], index=3, horizontal=True)
    fin = dolar.index.max()
    if periodo == "Personalizado":
        rango = st.date_input(
            "Fechas",
            value=(dolar.index.min().date(), fin.date()),
            min_value=dolar.index.min().date(),
            max_value=fin.date(),
        )
        if isinstance(rango, tuple) and len(rango) == 2:
            inicio, fin = pd.Timestamp(rango[0]), pd.Timestamp(rango[1])
        else:
            inicio = dolar.index.min()
    elif PERIODOS[periodo] is None:
        inicio = dolar.index.min()
    else:
        inicio = fin - pd.Timedelta(days=PERIODOS[periodo])
    seleccion = st.multiselect("Cotizaciones", list(dolar.columns), default=list(dolar.columns))
    st.caption("Fuente: argentinadatos.com (BCRA, INDEC y mercado)")

d = dolar.loc[inicio:fin]
infl_f = infl.loc[inicio:fin]
if len(infl_f) < 6:
    infl_f = infl.tail(6)

# Header: cotizaciones con variación a 30 días
st.caption(f"Datos al {dolar.index[-1]:%d/%m/%Y} · variaciones contra hace 30 días")
cols = st.columns(len(dolar.columns))
for c, name in zip(cols, dolar.columns):
    delta = change(dolar[name])
    c.metric(f"Dólar {name}", ars(dolar[name].iloc[-1]),
             f"{delta:+.1f}%" if delta is not None else None, delta_color="off")

# Header: brechas e inflación
brecha_cols = [n for n in ["Blue", "MEP", "CCL"] if n in dolar.columns]
r2 = st.columns(len(brecha_cols) + 2)
for c, name in zip(r2, brecha_cols):
    b = (dolar[name] / dolar["Oficial"] - 1) * 100
    delta = change(b, pct=False)
    c.metric(f"Brecha {name}", f"{b.iloc[-1]:.1f}%",
             f"{delta:+.1f} p.p." if delta is not None else None, delta_color="inverse")
m, ia = infl["mensual"], infl["interanual"].dropna()
r2[-2].metric(f"Inflación {MESES[infl.index[-1].month - 1]} {infl.index[-1].year}",
              f"{m.iloc[-1]:.1f}%", f"{m.iloc[-1] - m.iloc[-2]:+.1f} p.p. vs mes ant.",
              delta_color="inverse")
r2[-1].metric("Inflación interanual", f"{ia.iloc[-1]:.1f}%",
              f"{ia.iloc[-1] - ia.iloc[-2]:+.1f} p.p. vs mes ant.", delta_color="inverse")

tab1, tab2, tab3, tab4 = st.tabs(["💵 Dólar", "🛒 Inflación", "🧮 Calculadora", "🤖 Modelo"])

with tab1:
    if not seleccion:
        st.info("Elegí al menos una cotización en la barra lateral.")
    else:
        st.plotly_chart(px.line(d[seleccion], labels={"value": "ARS", "fecha": "", "variable": ""}),
                        width="stretch")
        otras = [n for n in seleccion if n != "Oficial"]
        if otras:
            brechas = d[otras].div(d["Oficial"], axis=0).sub(1).mul(100)
            st.subheader("Brecha contra el oficial")
            st.plotly_chart(px.line(brechas, labels={"value": "Brecha (%)", "fecha": "", "variable": ""}),
                            width="stretch")
        st.subheader(f"Resumen del período ({inicio:%d/%m/%Y} – {fin:%d/%m/%Y})")
        resumen = pd.DataFrame({"Mínimo": d[seleccion].min(), "Promedio": d[seleccion].mean(),
                                "Máximo": d[seleccion].max(), "Último": d[seleccion].iloc[-1]})
        st.dataframe(resumen.round(0), width="stretch")

with tab2:
    st.plotly_chart(px.bar(infl_f, y="mensual", labels={"mensual": "Inflación mensual (%)", "fecha": ""}),
                    width="stretch")
    st.plotly_chart(px.line(infl_f.dropna(), y="interanual",
                            labels={"interanual": "Inflación interanual (%)", "fecha": ""}),
                    width="stretch")

with tab3:
    st.subheader("Calculadora de inflación")
    st.write("¿Cuánto equivale hoy un monto de otra época? Se aplica la inflación de cada mes "
             "posterior al mes de inicio, hasta el mes final inclusive.")
    labels = [f"{MESES[x.month - 1].capitalize()} {x.year}" for x in infl.index]
    k1, k2, k3 = st.columns(3)
    monto = k1.number_input("Monto ($)", min_value=0.0, value=100000.0, step=1000.0)
    i0 = k2.selectbox("Desde", range(len(labels)), index=max(len(labels) - 13, 0),
                      format_func=lambda i: labels[i])
    i1 = k3.selectbox("Hasta", range(len(labels)), index=len(labels) - 1,
                      format_func=lambda i: labels[i])
    if i1 <= i0:
        st.warning("El mes final tiene que ser posterior al mes inicial.")
    else:
        tramo = infl["mensual"].iloc[i0 + 1:i1 + 1]
        acum = (1 + tramo / 100).cumprod()
        factor = acum.iloc[-1]
        k1, k2, k3 = st.columns(3)
        k1.metric(f"{ars(monto)} de {labels[i0]} equivalen a", ars(monto * factor))
        k2.metric("Inflación acumulada", f"{(factor - 1) * 100:.1f}%")
        k3.metric("Poder de compra perdido", f"{(1 - 1 / factor) * 100:.1f}%")
        serie = pd.concat([pd.Series([monto], index=[infl.index[i0]]), acum * monto])
        st.plotly_chart(px.line(serie.rename("Monto ajustado"),
                                labels={"value": "ARS", "index": ""}), width="stretch")

with tab4:
    st.subheader("Predicción de la inflación mensual")
    st.write("Regresión Ridge con 6 rezagos + media móvil de 3 meses, validada con "
             "*walk-forward* sobre los últimos 12 meses y comparada con un baseline ingenuo "
             "(repetir el mes anterior).")
    result, metrics, next_pred = backtest_and_forecast(infl["mensual"])
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE modelo (p.p.)", f"{metrics['MAE modelo']:.2f}")
    m2.metric("MAE ingenuo (p.p.)", f"{metrics['MAE ingenuo']:.2f}")
    m3.metric("Próximo mes (estimado)", f"{next_pred:.1f}%")
    st.plotly_chart(px.line(result, y=["real", "modelo", "ingenuo"],
                            labels={"value": "Inflación mensual (%)", "fecha": "", "variable": ""}),
                    width="stretch")
    st.info("Con series cortas y cambios de régimen, el baseline ingenuo suele ser difícil de "
            "superar. Reportarlo con honestidad es parte del análisis.")
