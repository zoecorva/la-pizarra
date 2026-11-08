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


@st.cache_data(ttl=3600)
def load_plazo_fijo() -> pd.DataFrame:
    df = pd.DataFrame(get_json("finanzas/tasas/plazoFijo"))
    df = df.reindex(columns=["entidad", "tnaClientes", "tnaNoClientes"])
    for c in ["tnaClientes", "tnaNoClientes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[["tnaClientes", "tnaNoClientes"]].max().max() <= 2:  # vino como fracción (0.30) y no como %
        df[["tnaClientes", "tnaNoClientes"]] *= 100
    return df.dropna(subset=["entidad"]).dropna(subset=["tnaClientes", "tnaNoClientes"], how="all")


# ---------- BCRA (API oficial, v4.0) ----------
BCRA_URL = "https://api.bcra.gob.ar/estadisticas/v4.0/Monetarias"


@st.cache_data(ttl=3600)
def bcra_catalogo() -> pd.DataFrame:
    r = requests.get(BCRA_URL, params={"categoria": "Principales Variables", "limit": 1000}, timeout=30)
    r.raise_for_status()
    cols = ["idVariable", "descripcion", "periodicidad", "unidadExpresion"]
    return pd.DataFrame(r.json()["results"]).reindex(columns=cols)


@st.cache_data(ttl=3600)
def bcra_serie(id_variable: int, desde: str, hasta: str) -> pd.Series:
    rows, offset = [], 0
    while True:
        r = requests.get(f"{BCRA_URL}/{id_variable}", timeout=30,
                         params={"desde": desde, "hasta": hasta, "limit": 3000, "offset": offset})
        r.raise_for_status()
        j = r.json()
        detalle = j["results"][0]["detalle"] if j.get("results") else []
        rows += detalle
        offset += 3000
        if not detalle or offset >= j["metadata"]["resultset"]["count"]:
            break
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.drop_duplicates("fecha").set_index("fecha")["valor"].sort_index()


# ---------- Modelo ----------
LAGS = [1, 2, 3, 6]


def make_features(s: pd.Series) -> pd.DataFrame:
    X = pd.concat({f"lag_{i}": s.shift(i) for i in LAGS}, axis=1)
    X["media_3m"] = s.shift(1).rolling(3).mean()
    X["media_6m"] = s.shift(1).rolling(6).mean()
    return X


@st.cache_data(ttl=3600)
def backtest_and_forecast(s: pd.Series, test_size: int = 12):
    """Walk-forward: reentrena mes a mes y compara modelo, baseline ingenuo y ensemble 50/50."""
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
    # Pesos fijos (50/50): no se ajustan mirando el período de prueba, así no hay fuga de información
    result["ensemble"] = 0.5 * result["modelo"] + 0.5 * result["ingenuo"]
    metrics = {k: mean_absolute_error(result["real"], result[k])
               for k in ["modelo", "ingenuo", "ensemble"]}

    # Pronóstico del próximo mes con el modelo entrenado sobre todo el historial
    final = Ridge(alpha=1.0).fit(data[feats], data["y"])
    last = {f"lag_{i}": s.iloc[-i] for i in LAGS}
    last["media_3m"] = s.iloc[-3:].mean()
    last["media_6m"] = s.iloc[-6:].mean()
    modelo_next = final.predict(pd.DataFrame([last])[feats])[0]
    forecast = {"modelo": modelo_next, "ensemble": 0.5 * modelo_next + 0.5 * s.iloc[-1]}
    return result, metrics, forecast


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
    st.caption("Fuentes: argentinadatos.com y API del BCRA")

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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["💵 Dólar", "🛒 Inflación", "🧮 Calculadora", "🏦 BCRA", "🏛️ Bancos", "🤖 Modelo"])

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
    st.subheader("Datos directos del BCRA")
    st.write("Series oficiales de las *Principales Variables* del Banco Central, "
             "consultadas en vivo a su API. Elegí una variable para ver su evolución en el período seleccionado.")
    try:
        cat = bcra_catalogo().dropna(subset=["idVariable", "descripcion"])
        nombres = cat.set_index("idVariable")["descripcion"]
        ids = list(nombres.index)
        vid = st.selectbox("Variable", ids, index=ids.index(1) if 1 in ids else 0,
                           format_func=lambda i: nombres[i])
        fila = cat[cat["idVariable"] == vid].iloc[0]
        serie = bcra_serie(int(vid), f"{inicio:%Y-%m-%d}", f"{fin:%Y-%m-%d}")
    except (requests.RequestException, KeyError, ValueError, IndexError) as e:
        st.warning(f"No pude consultar la API del BCRA en este momento ({e}). El resto del dashboard sigue funcionando.")
    else:
        if serie.empty:
            st.info("Esa variable no tiene datos en el período elegido. Probá con un período más largo.")
        else:
            unidad = fila["unidadExpresion"] if pd.notna(fila["unidadExpresion"]) else ""
            k1, k2, k3 = st.columns(3)
            delta = (serie.iloc[-1] / serie.iloc[0] - 1) * 100 if serie.iloc[0] else None
            k1.metric("Último valor", f"{serie.iloc[-1]:,.2f}", f"{delta:+.1f}% en el período" if delta is not None else None,
                      delta_color="off")
            k2.metric("Fecha", f"{serie.index[-1]:%d/%m/%Y}")
            k3.metric("Unidad", unidad or "—")
            st.plotly_chart(px.line(serie.rename(nombres[vid]), labels={"value": unidad, "fecha": "", "variable": ""}),
                            width="stretch")
    st.caption("Fuente: API de Estadísticas Monetarias v4.0 del BCRA (api.bcra.gob.ar).")

with tab5:
    st.subheader("Plazo fijo: comparación entre bancos")
    st.write("Tasas que cada banco informó al BCRA para plazos fijos online de $100.000 a 30 días. "
             "Se compara el rendimiento contra la inflación del último mes.")
    try:
        pf = load_plazo_fijo()
    except (requests.RequestException, KeyError, ValueError) as e:
        st.warning(f"No pude cargar las tasas de los bancos ({e}). El resto del dashboard sigue funcionando.")
    else:
        b1, b2 = st.columns(2)
        tipo = b1.radio("Tasa para", ["Clientes", "No clientes"], horizontal=True)
        monto_pf = b2.number_input("Monto a invertir ($)", min_value=0.0, value=1_000_000.0, step=100_000.0)
        col_tna = "tnaClientes" if tipo == "Clientes" else "tnaNoClientes"
        t = pf[["entidad", col_tna]].dropna().rename(columns={col_tna: "TNA"})
        t["Interés a 30 días"] = monto_pf * t["TNA"] / 100 * 30 / 365
        t["Rendimiento real vs inflación (%)"] = ((1 + t["TNA"] / 100 * 30 / 365) / (1 + infl["mensual"].iloc[-1] / 100) - 1) * 100
        t = t.sort_values("TNA", ascending=False).reset_index(drop=True)
        if t.empty:
            st.info("No hay tasas disponibles para esa selección.")
        else:
            gana = int((t["Rendimiento real vs inflación (%)"] > 0).sum())
            k1, k2, k3 = st.columns(3)
            k1.metric("Mejor tasa", f"{t['TNA'].iloc[0]:.1f}%", t["entidad"].iloc[0], delta_color="off")
            k2.metric("Promedio de bancos", f"{t['TNA'].mean():.1f}%")
            k3.metric("Le ganan a la inflación", f"{gana} de {len(t)}",
                      f"inflación último mes: {infl['mensual'].iloc[-1]:.1f}%", delta_color="off")
            st.plotly_chart(px.bar(t.sort_values("TNA"), x="TNA", y="entidad", orientation="h",
                                   labels={"TNA": "TNA (%)", "entidad": ""}, height=max(400, 22 * len(t))),
                            width="stretch")
            st.dataframe(t.round(2), width="stretch", hide_index=True)
    st.caption("Fuente: Régimen de Transparencia del BCRA, vía argentinadatos.com. "
               "La tasa puede cambiar para otros montos o plazos. Dato informativo, no es una recomendación de inversión.")

with tab6:
    st.subheader("Predicción de la inflación mensual")
    st.write("Regresión Ridge con rezagos de 1, 2, 3 y 6 meses y medias móviles de 3 y 6 meses, "
             "validada con *walk-forward* sobre los últimos 12 meses. Se compara contra un baseline "
             "ingenuo (repetir el mes anterior) y contra un ensemble 50/50 de ambos.")
    result, metrics, forecast = backtest_and_forecast(infl["mensual"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE modelo (p.p.)", f"{metrics['modelo']:.2f}")
    m2.metric("MAE ingenuo (p.p.)", f"{metrics['ingenuo']:.2f}")
    m3.metric("MAE ensemble (p.p.)", f"{metrics['ensemble']:.2f}")
    m4.metric("Próximo mes (ensemble)", f"{forecast['ensemble']:.1f}%",
              f"modelo solo: {forecast['modelo']:.1f}%", delta_color="off")
    st.plotly_chart(px.line(result, y=["real", "modelo", "ingenuo", "ensemble"],
                            labels={"value": "Inflación mensual (%)", "fecha": "", "variable": ""}),
                    width="stretch")
    st.info("Los pesos del ensemble son fijos (50/50) y no se ajustan mirando el período de prueba, "
            "para no inflar el resultado. Con series cortas y cambios de régimen, el baseline "
            "ingenuo suele ser difícil de superar: reportarlo con honestidad es parte del análisis.")
    
