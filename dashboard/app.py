"""Dashboard del certamen: evaluar probabilidades exportadas, sin entrenar."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


CARPETA = Path(__file__).resolve().parent
st.set_page_config(page_title="Dashboard Certamen 1", layout="wide")


@st.cache_data
# Las rutas dependen del archivo y no del directorio de ejecución.
def cargar_datos():
    datos = pd.read_csv(CARPETA / "datos_dashboard.csv")
    config = json.loads((CARPETA / "config.json").read_text(encoding="utf-8"))
    requeridas = {"y_real", "probabilidad_clase_1"}
    if not requeridas.issubset(datos.columns) or datos.empty:
        raise ValueError("El CSV debe contener y_real y probabilidad_clase_1, sin estar vacío.")
    if not datos["y_real"].isin([0, 1]).all():
        raise ValueError("y_real debe contener únicamente 0 y 1, sin nulos.")
    if not datos["probabilidad_clase_1"].between(0, 1).all():
        raise ValueError("Las probabilidades deben estar entre 0 y 1, sin nulos.")
    claves = ["C_FP", "C_FN", "umbral_teorico", "umbral_empirico_manual", "umbral_tuned"]
    for clave in claves:
        if clave not in config or not np.isfinite(float(config[clave])):
            raise ValueError(f"Falta un valor válido para {clave} en config.json.")
    if min(config["C_FP"], config["C_FN"]) < 0:
        raise ValueError("Los costos no pueden ser negativos.")
    for clave in claves[2:]:
        if not 0 <= config[clave] <= 1:
            raise ValueError(f"{clave} debe estar entre 0 y 1.")
    return datos, config


def evaluar(y_real, probabilidades, umbral, costo_fp, costo_fn):
    """Recalcular errores y costos a partir de las probabilidades guardadas."""
    pred = (probabilidades >= umbral).astype(int)
    tn = int(((y_real == 0) & (pred == 0)).sum())
    fp = int(((y_real == 0) & (pred == 1)).sum())
    fn = int(((y_real == 1) & (pred == 0)).sum())
    tp = int(((y_real == 1) & (pred == 1)).sum())
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "Precision": tp / (tp + fp) if tp + fp else 0.0,
        "Recall": tp / (tp + fn) if tp + fn else 0.0,
        "Costo": fp * costo_fp + fn * costo_fn,
    }


def dinero(valor):
    signo = "-" if valor < 0 else ""
    return signo + "$" + f"{abs(valor):,.0f}".replace(",", ".")


def agrupar_umbrales(politicas):
    """Unir nombres de políticas que usan el mismo umbral."""
    grupos = {}
    for nombre, umbral in politicas:
        clave = round(float(umbral), 12)
        if clave not in grupos:
            grupos[clave] = {"Umbral": float(umbral), "Politica": nombre}
        else:
            grupos[clave]["Politica"] += " / " + nombre
    return list(grupos.values())


st.title("Dashboard de Costos y Umbral de Clasificación")
st.caption("Porto Seguro Safe Driver Prediction")
st.write(
    "El dashboard muestra cómo cambian los errores de clasificación y su costo económico "
    "al modificar los costos de FP/FN y el umbral de decisión."
)
try:
    datos, config = cargar_datos()
except (OSError, ValueError, TypeError) as error:
    st.error(f"No se pudieron cargar los datos del dashboard: {error}")
    st.stop()

y_real = datos["y_real"].to_numpy(dtype=int)
probabilidades = datos["probabilidad_clase_1"].to_numpy(dtype=float)

with st.sidebar:
    st.header("Costos y umbral")
    C_FP = st.number_input(
        "Costo Falso Positivo (C_FP)", min_value=0,
        value=int(config["C_FP"]), step=100, key="C_FP",
    )
    C_FN = st.number_input(
        "Costo Falso Negativo (C_FN)", min_value=0,
        value=int(config["C_FN"]), step=100, key="C_FN",
    )
    if C_FP + C_FN == 0:
        st.warning("Al menos uno de los costos debe ser mayor que cero para definir el umbral teórico.")
        st.stop()
    p_teorico = C_FP / (C_FP + C_FN)
    st.metric("Umbral óptimo teórico", f"{p_teorico:.4f}")
    umbral_inicial = float(np.clip(round(config["umbral_tuned"], 2), 0.01, 0.99))
    umbral = st.slider(
        "Umbral de clasificación", min_value=0.01, max_value=0.99,
        value=umbral_inicial, step=0.01, key="umbral",
    )
    st.caption(
        f"Seleccionado: **{umbral:.2f}** · Teórico actual: **{p_teorico:.4f}**  \n"
        f"Tuned original del trabajo: **{config['umbral_tuned']:.2f}**  \n"
        f"Empírico manual original: **{config['umbral_empirico_manual']:.2f}**"
    )
    st.caption("Costos académicos de negocio; no representan costos reales de Porto Seguro.")

actual = evaluar(y_real, probabilidades, umbral, C_FP, C_FN)
base = evaluar(y_real, probabilidades, 0.5, C_FP, C_FN)
ahorro = base["Costo"] - actual["Costo"]

indicadores = st.columns(4)
indicadores[0].metric("Recall", f"{actual['Recall']:.2%}")
indicadores[1].metric("Precision", f"{actual['Precision']:.2%}")
indicadores[2].metric("Costo total", dinero(actual["Costo"]))
indicadores[3].metric("Ahorro vs umbral 0.5", dinero(ahorro))
st.caption(f"Evaluación sobre {len(datos):,} observaciones de Test. Costo con umbral 0.5: {dinero(base['Costo'])}.".replace(",", "."))

izquierda, derecha = st.columns([1, 1.5])
with izquierda:
    st.subheader("Matriz de confusión")
    matriz = [[actual["TN"], actual["FP"]], [actual["FN"], actual["TP"]]]
    texto = [[f"TN<br>{actual['TN']:,}", f"FP<br>{actual['FP']:,}"],
             [f"FN<br>{actual['FN']:,}", f"TP<br>{actual['TP']:,}"]]
    fig_matriz = go.Figure(go.Heatmap(
        z=matriz, x=["Predicho 0", "Predicho 1"], y=["Real 0", "Real 1"],
        text=texto, texttemplate="%{text}", colorscale="Blues", showscale=False,
        hovertemplate="%{y} · %{x}<br>Registros: %{z:,}<extra></extra>",
    ))
    fig_matriz.update_layout(height=330, margin=dict(l=10, r=10, t=15, b=10))
    fig_matriz.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_matriz, width="stretch", key="matriz")
    # Escapar $ evita que Markdown interprete los importes como fórmulas.
    st.write(
        f"Costo unitario FP: {dinero(C_FP)} | Costo unitario FN: {dinero(C_FN)}".replace("$", r"\$")
    )
    st.write(
        f"Costo por FP: {dinero(actual['FP'] * C_FP)} | "
        f"Costo por FN: {dinero(actual['FN'] * C_FN)}".replace("$", r"\$")
    )

with derecha:
    st.subheader("Costo total según umbral")
    grilla = np.arange(1, 100) / 100
    costos = [evaluar(y_real, probabilidades, t, C_FP, C_FN)["Costo"] for t in grilla]
    fig_costos = go.Figure(go.Scatter(
        x=grilla, y=costos, mode="lines", name="Costo en Test",
        line=dict(color="#2563a6", width=3),
        hovertemplate="Umbral: %{x:.2f}<br>Costo: $%{y:,.0f}<extra></extra>",
    ))
    politicas = [("0.5", 0.5), ("Teórico actual", p_teorico),
                 ("Seleccionado por usuario", umbral), ("Tuned original", config["umbral_tuned"])]
    colores = ["#777777", "#d77b12", "#16836b", "#8c57a6"]
    for politica, color in zip(agrupar_umbrales(politicas), colores):
        fig_costos.add_vline(
            x=politica["Umbral"], line_dash="dash", line_color=color,
            name=politica["Politica"], showlegend=True,
        )
    fig_costos.update_layout(
        height=370, margin=dict(l=10, r=10, t=15, b=10),
        xaxis_title="Umbral", yaxis_title="Costo total", yaxis_tickprefix="$",
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig_costos, width="stretch", key="curva_costos")
    st.caption(
        "Esta curva corresponde a la evaluación sobre Test y se utiliza únicamente para "
        "visualizar el efecto económico. No reentrena ni selecciona el modelo, ni modifica "
        "los umbrales originales elegidos con Train."
    )

st.subheader("Comparación de umbrales")
filas = []
for politica in agrupar_umbrales(politicas):
    metricas = evaluar(y_real, probabilidades, politica["Umbral"], C_FP, C_FN)
    filas.append({**politica, **{c: metricas[c] for c in ["Precision", "Recall", "Costo"]}})
comparacion = pd.DataFrame(filas)
st.dataframe(
    comparacion.style.format({"Umbral": "{:.4f}", "Precision": "{:.2%}",
                              "Recall": "{:.2%}", "Costo": dinero}),
    hide_index=True, width="stretch",
)

st.subheader("Interpretación")
comparacion_recall = "mayor" if actual["Recall"] > base["Recall"] else "igual"
comparacion_fp = "más" if actual["FP"] > base["FP"] else "la misma cantidad de"
st.write(
    f"Con el umbral seleccionado ({umbral:.2f}), el recall es {comparacion_recall} que con 0.5 "
    f"y se obtienen {comparacion_fp} falsos positivos ({actual['FP']:,} frente a {base['FP']:,}). "
    "Al bajar el umbral normalmente se detectan más positivos y aumenta el recall; "
    "también pueden aumentar los falsos positivos."
)
st.write(
    f"La relación actual C_FN/C_FP es {C_FN / C_FP:.1f}. "
    if C_FP else "El costo de FP es cero. "
)
st.write(
    "El umbral económico depende de la relación entre los costos de FN y FP. "
    "Un ahorro negativo indica un costo mayor que con 0.5. "
    "Estas comparaciones describen los datos de Test y no demuestran relaciones causales."
)
