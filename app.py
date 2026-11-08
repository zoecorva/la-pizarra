"""La Pizarra, dashboard macroeconómico de Argentina: dólares, brechas, inflación, calculadora y modelo."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

BASE_URL = "https://api.argentinadatos.com/v1"
CASAS = {"oficial": "Oficial", "blue": "Blue", "bolsa": "MEP", "contadoconliqui": "CCL"}
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
PERIODOS = {"1 mes": 30, "3 meses": 90, "6 meses": 180, "1 año": 365, "3 años": 1095, "Todo": None}

st.set_page_config(page_title="La Pizarra", page_icon="📈", layout="wide")

# ---------- Identidad visual: papel moneda ----------
TINTA, PAPEL, ORO, VERDE, ROJO = "#14213D", "#EEF2F1", "#E0A526", "#2F6B4F", "#B23A48"
COLORES = {"Oficial": TINTA, "Blue": ORO, "MEP": VERDE, "CCL": ROJO}

pio.templates["macro"] = go.layout.Template(layout=dict(
    font=dict(family="Hanken Grotesk, sans-serif", color=TINTA, size=13),
    colorway=[TINTA, ORO, VERDE, ROJO, "#6B7A99", "#9A7B4F"],
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="rgba(20,33,61,.07)", linecolor=TINTA, zeroline=False, title_text=""),
    yaxis=dict(gridcolor="rgba(20,33,61,.07)", zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
    margin=dict(l=0, r=0, t=36, b=0),
))
pio.templates.default = "macro"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Hanken+Grotesk:wght@400;500;700&display=swap');
.stApp { background: #EEF2F1; }
.stApp, .stApp p, .stApp label, .stApp li, [data-testid="stMarkdownContainer"] *, [data-testid="stCaptionContainer"] * {
    font-family: 'Hanken Grotesk', sans-serif; color: #14213D; }
[data-testid="stSidebar"] { background: #E1E8E6; }
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
footer { visibility: hidden; }

/* Portada: guilloché de billete hecho con círculos concéntricos */
.hero { position: relative; overflow: hidden; border-radius: 6px; padding: 2.6rem 2.4rem 2.3rem; margin-bottom: 1.8rem; background: #14213D; }
.hero::before { content: ""; position: absolute; inset: 0; background:
    repeating-radial-gradient(circle at 88% 18%, transparent 0 7px, rgba(238,242,241,.10) 7px 8px),
    repeating-radial-gradient(circle at 72% 95%, transparent 0 11px, rgba(224,165,38,.16) 11px 12px); }
.hero > div { position: relative; }
.hero .hero-title { font-family: 'Newsreader', serif !important; font-weight: 500; color: #EEF2F1 !important;
    font-size: clamp(2.8rem, 7vw, 5rem); line-height: .98; letter-spacing: -.025em; }
.hero .hero-sub { max-width: 32rem; margin-top: 1rem; font-size: 1.08rem; line-height: 1.5; color: #EEF2F1 !important; opacity: .88; }
.hero .hero-date { margin-top: 1.4rem; font-size: .9rem; color: #E0A526 !important; font-weight: 500; }


/* Tablero: baldosas con número, variación y minigráfico */
.grid { display: grid; gap: .7rem; margin: 0 0 1.3rem; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); }
.tile { border-radius: 8px; padding: .85rem 1rem .6rem; overflow: hidden; }
.tile.dark { background: #14213D; }
.tile.light { background: #E1E8E6; }
.tile.dark, .tile.dark * { color: #EEF2F1 !important; }
.tile.light, .tile.light * { color: #14213D !important; }
.t-label { font-size: .86rem; opacity: .8; }
.t-value { font-family: 'Newsreader', serif !important; font-size: 2.1rem; font-weight: 500; line-height: 1.1; margin: .1rem 0 .4rem; letter-spacing: -.01em; }
.t-value.long { font-size: 1.2rem; line-height: 1.3; padding: .35rem 0 .25rem; }
.t-sub { font-size: .8rem; opacity: .75; }
.chip { display: inline-block; font-size: .78rem; font-weight: 700; padding: .08rem .5rem; border-radius: 99px; }
.tile.dark .chip.up { background: rgba(240,138,150,.22); color: #FFC4CB !important; }
.tile.dark .chip.down { background: rgba(111,191,154,.22); color: #A9E5C7 !important; }
.tile.light .chip.up { background: rgba(178,58,72,.14); color: #B23A48 !important; }
.tile.light .chip.down { background: rgba(47,107,79,.16); color: #2F6B4F !important; }
.chip.flat { background: rgba(128,128,128,.2); }
.spark { display: block; width: 100%; height: 30px; margin-top: .55rem; }
.lead { font-family: 'Newsreader', serif; font-size: 1.28rem; line-height: 1.4; margin: 0 0 1.2rem; max-width: 46rem; }
.lead, .lead * { font-family: 'Newsreader', serif !important; }

/* Indicadores: números grandes de imprenta, sin tarjetas */
[data-testid="stMetric"] { background: transparent; border-top: 2px solid #14213D; padding: .55rem 0 .4rem; }
[data-testid="stMetricLabel"] * { font-size: .92rem; color: #4A5568 !important; }
[data-testid="stMetricValue"] * { font-family: 'Newsreader', serif !important; font-weight: 500; font-size: 2.25rem; letter-spacing: -.01em; }
[data-testid="stMetricDelta"] * { font-size: .85rem; }

/* Pestañas */
[data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid rgba(20,33,61,.18); }
[data-baseweb="tab"] { padding-left: 0; padding-right: 0; }
[data-baseweb="tab"] p { font-size: 1.02rem; font-weight: 500; }
[data-baseweb="tab-highlight"] { background-color: #E0A526 !important; height: 3px; }

/* Títulos */
h2, h3, [data-testid="stHeading"] * { font-family: 'Newsreader', serif !important; font-weight: 500; letter-spacing: -.01em; color: #14213D !important; }
.stAlert { border-radius: 4px; }
@media (max-width: 640px) {
    div.block-container, [data-testid="stMainBlockContainer"] { padding: .8rem .85rem 3rem !important; }
    [data-testid="stVerticalBlock"] { gap: .8rem !important; }
    .hero { padding: 1.5rem 1.2rem; margin-bottom: 1rem; }
    .grid { grid-template-columns: repeat(2, 1fr); gap: .55rem; }
    .t-value { font-size: 1.75rem; }
    .lead { font-size: 1.12rem; }
    [data-testid="stHeading"] h3 { font-size: 1.45rem; padding: .2rem 0 0; }
    [data-baseweb="tab-list"] { gap: 1rem; }
    [data-baseweb="tab"] p { font-size: .95rem; }
}
</style>
""", unsafe_allow_html=True)


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


def num(x: float, d: int = 1) -> str:
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def chart(fig, fecha: bool = True, **_):
    """Gráfico sin barra de herramientas ni zoom táctil (en el celu se activaba al hacer scroll)."""
    fig.update_layout(dragmode=False, height=fig.layout.height or 320)
    if fecha:
        fig.update_xaxes(tickformat="%m/%y", hoverformat="%d/%m/%Y")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "scrollZoom": False})


def spark(series: pd.Series, color: str, n: int = 45) -> str:
    v = series.dropna().tail(n).to_numpy(dtype=float)
    if len(v) < 2:
        return ""
    lo, span = v.min(), (v.max() - v.min()) or 1.0
    pts = " ".join(f"{i * 100 / (len(v) - 1):.1f},{28 - (x - lo) / span * 25 - 1.5:.1f}" for i, x in enumerate(v))
    return (f'<svg class="spark" viewBox="0 0 100 28" preserveAspectRatio="none"><polyline points="{pts}" '
            f'fill="none" stroke="{color}" stroke-width="2" vector-effect="non-scaling-stroke" '
            f'stroke-linejoin="round"/></svg>')


def chip(x, suf: str) -> str:
    if x is None:
        return ""
    cls = "up" if x > 0.05 else "down" if x < -0.05 else "flat"
    arrow = "▲" if cls == "up" else "▼" if cls == "down" else "■"
    return f'<span class="chip {cls}">{arrow} {num(abs(x))}{suf}</span>'


def tile(label: str, value: str, foot: str = "", spark_svg: str = "", kind: str = "light") -> str:
    return (f'<div class="tile {kind}"><div class="t-label">{label}</div><div class="t-value{" long" if len(value) > 14 else ""}">{value}</div>'
            f'<div>{foot}</div>{spark_svg}</div>')


def board(tiles: list) -> None:
    st.markdown(f'<div class="grid">{"".join(tiles)}</div>', unsafe_allow_html=True)


def kpis(items: list) -> None:
    """items: lista de (etiqueta, valor, texto_chico_opcional)."""
    board([tile(a, b, f'<span class="t-sub">{c}</span>' if c else "") for a, b, c, *_ in items])



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
try:
    dolar, infl = load_dolar(), load_inflacion()
except requests.RequestException as e:
    st.error(f"No pude descargar los datos: {e}")
    st.stop()


st.markdown(f"""
<div class="hero"><div>
  <div class="hero-title">La Pizarra</div>
  <div class="hero-sub">Dólar, inflación y tasas de Argentina en un solo tablero, con datos oficiales.</div>
  <div class="hero-date">Datos al {dolar.index[-1]:%d/%m/%Y}. Se actualizan cada hora.</div>
</div></div>
""", unsafe_allow_html=True)

# Sidebar: período y cotizaciones
with st.expander("Filtros: período y cotizaciones"):
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

d = dolar.loc[inicio:fin]
infl_f = infl.loc[inicio:fin]
if len(infl_f) < 6:
    infl_f = infl.tail(6)

# Tablero principal
m, ia = infl["mensual"], infl["interanual"].dropna()
mes_txt = f"{MESES[infl.index[-1].month - 1]} {infl.index[-1].year}"
ref = "CCL" if "CCL" in dolar.columns else "Blue" if "Blue" in dolar.columns else None
if ref:
    br = (dolar[ref].iloc[-1] / dolar["Oficial"].iloc[-1] - 1) * 100
    st.markdown(f'<p class="lead">El dólar {ref} cotiza {num(abs(br))}% {"por encima" if br >= 0 else "por debajo"} del oficial. '
                f'La inflación de {mes_txt} fue {num(m.iloc[-1])}% y la interanual, {num(ia.iloc[-1])}%.</p>',
                unsafe_allow_html=True)

SPARK_OSCURO = {"Oficial": "#EEF2F1", "Blue": ORO, "MEP": "#6FBF9A", "CCL": "#F08A96"}
board([tile(f"Dólar {n}", ars(dolar[n].iloc[-1]), chip(change(dolar[n]), "%"),
            spark(dolar[n], SPARK_OSCURO.get(n, ORO)), "dark") for n in dolar.columns])
st.caption("Variación contra hace 30 días. El minigráfico muestra las últimas semanas.")

tiles = []
for n in [x for x in ["Blue", "MEP", "CCL"] if x in dolar.columns]:
    bs = (dolar[n] / dolar["Oficial"] - 1) * 100
    tiles.append(tile(f"Brecha {n}", f"{num(bs.iloc[-1])}%", chip(change(bs, pct=False), " p.p."),
                      spark(bs, COLORES[n])))
tiles.append(tile(f"Inflación {mes_txt}", f"{num(m.iloc[-1])}%", chip(m.iloc[-1] - m.iloc[-2], " p.p."),
                  spark(m, TINTA, 12)))
tiles.append(tile("Inflación interanual", f"{num(ia.iloc[-1])}%", chip(ia.iloc[-1] - ia.iloc[-2], " p.p."),
                  spark(ia, TINTA, 24)))
board(tiles)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Dólar", "Inflación", "Calculadora", "BCRA", "Bancos", "Modelo"])

with tab1:
    if not seleccion:
        st.info("Elegí al menos una cotización en la barra lateral.")
    else:
        chart(px.line(d[seleccion], color_discrete_map=COLORES, labels={"value": "ARS", "fecha": "", "variable": ""}),
                        width="stretch")
        otras = [n for n in seleccion if n != "Oficial"]
        if otras:
            brechas = d[otras].div(d["Oficial"], axis=0).sub(1).mul(100)
            st.subheader("Brecha contra el oficial")
            chart(px.line(brechas, color_discrete_map=COLORES, labels={"value": "Brecha (%)", "fecha": "", "variable": ""}),
                            width="stretch")
        st.subheader("Resumen del período")
        st.caption(f"Del {inicio:%d/%m/%Y} al {fin:%d/%m/%Y}")
        resumen = pd.DataFrame({"Mínimo": d[seleccion].min(), "Promedio": d[seleccion].mean(),
                                "Máximo": d[seleccion].max(), "Último": d[seleccion].iloc[-1]})
        st.dataframe(resumen.round(0), width="stretch")

with tab2:
    chart(px.bar(infl_f, y="mensual", labels={"mensual": "Inflación mensual (%)", "fecha": ""}),
                    width="stretch")
    chart(px.line(infl_f.dropna(), y="interanual",
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
        kpis([(f"{ars(monto)} de {labels[i0]} equivalen a", ars(monto * factor), ""),
              ("Inflación acumulada", f"{num((factor - 1) * 100)}%", ""),
              ("Poder de compra perdido", f"{num((1 - 1 / factor) * 100)}%", "")])
        serie = pd.concat([pd.Series([monto], index=[infl.index[i0]]), acum * monto])
        chart(px.line(serie.rename("Monto ajustado"),
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
            delta = (serie.iloc[-1] / serie.iloc[0] - 1) * 100 if serie.iloc[0] else None
            kpis([("Último valor", num(serie.iloc[-1], 2), f"{num(delta)}% en el período" if delta is not None else ""),
                  ("Fecha", f"{serie.index[-1]:%d/%m/%Y}", ""),
                  ("Unidad", unidad or "—", "")])
            chart(px.line(serie.rename(nombres[vid]), labels={"value": unidad, "fecha": "", "variable": ""}),
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
            kpis([("Mejor tasa", f"{num(t['TNA'].iloc[0])}%", t["entidad"].iloc[0]),
                  ("Promedio de bancos", f"{num(t['TNA'].mean())}%", ""),
                  ("Le ganan a la inflación", f"{gana} de {len(t)}", f"inflación último mes: {num(infl['mensual'].iloc[-1])}%")])
            chart(px.bar(t.sort_values("TNA"), x="TNA", y="entidad", orientation="h",
                                   labels={"TNA": "TNA (%)", "entidad": ""}, height=max(400, 22 * len(t))),
                  fecha=False)
            st.dataframe(t.round(2), width="stretch", hide_index=True)
    st.caption("Fuente: Régimen de Transparencia del BCRA, vía argentinadatos.com. "
               "La tasa puede cambiar para otros montos o plazos. Dato informativo, no es una recomendación de inversión.")

with tab6:
    st.subheader("Predicción de la inflación mensual")
    st.write("Regresión Ridge con rezagos de 1, 2, 3 y 6 meses y medias móviles de 3 y 6 meses, "
             "validada con *walk-forward* sobre los últimos 12 meses. Se compara contra un baseline "
             "ingenuo (repetir el mes anterior) y contra un ensemble 50/50 de ambos.")
    result, metrics, forecast = backtest_and_forecast(infl["mensual"])
    kpis([("Error del modelo (p.p.)", num(metrics["modelo"], 2), ""),
          ("Error del ingenuo (p.p.)", num(metrics["ingenuo"], 2), ""),
          ("Error del ensemble (p.p.)", num(metrics["ensemble"], 2), ""),
          ("Próximo mes (ensemble)", f"{num(forecast['ensemble'])}%", f"modelo solo: {num(forecast['modelo'])}%")])
    chart(px.line(result, y=["real", "modelo", "ingenuo", "ensemble"],
                            labels={"value": "Inflación mensual (%)", "fecha": "", "variable": ""}),
                    width="stretch")
    st.info("Los pesos del ensemble son fijos (50/50) y no se ajustan mirando el período de prueba, "
            "para no inflar el resultado. Con series cortas y cambios de régimen, el baseline "
            "ingenuo suele ser difícil de superar: reportarlo con honestidad es parte del análisis.")

st.caption("Fuentes: INDEC y BCRA (a través de su API oficial) y argentinadatos.com.")
