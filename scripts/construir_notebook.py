"""Generar el notebook autocontenido del certamen (no ejecuta modelos)."""
from pathlib import Path
import nbformat

RAIZ = Path(__file__).resolve().parents[1]
celdas = []
def md(texto): celdas.append(nbformat.v4.new_markdown_cell(texto.strip()))
def codigo(texto): celdas.append(nbformat.v4.new_code_cell(texto.strip()))

md('''# 1. Problema de negocio y base de datos

El objetivo es predecir si un conductor presentará un reclamo/siniestro. `target=1` representa el reclamo y será la clase positiva. La predicción puede apoyar la evaluación de riesgo; los costos de error se definirán después como supuestos académicos.

Fuente: [Porto Seguro en OpenML, data_id=42742](https://www.openml.org/d/42742). Se verifica la clase minoritaria y los requisitos con los datos cargados, sin asumir los resultados de ejecuciones anteriores.
''')
codigo('''# Importar bibliotecas y localizar el proyecto.
from pathlib import Path
import sys
import json
import gzip
import cloudpickle
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

SEMILLA = 2026
RAIZ = Path.cwd()
if RAIZ.name == "notebooks":
    RAIZ = RAIZ.parent
assert (RAIZ / "scripts").is_dir(), "Ejecuta desde la raíz del proyecto o notebooks/."
RESULTADOS = RAIZ / "resultados"
RESULTADOS.mkdir(exist_ok=True)
sys.path.insert(0, str(RAIZ / "scripts"))
''')
codigo('''# Descargar automáticamente solo si no existe la copia local.
from descargar_porto_seguro import descargar_datos
ruta_csv = RAIZ / "data" / "raw" / "porto_seguro.csv"
if not ruta_csv.exists():
    descargar_datos()
df = pd.read_csv(ruta_csv)
objetivo = "target"
print("Shape:", df.shape)
print("Columnas:", df.columns.tolist())
print("Objetivo:", objetivo)
display(df.head())
conteo_clases = df[objetivo].value_counts().sort_index()
distribucion_clases = pd.DataFrame({
    "Cantidad": conteo_clases,
    "Porcentaje": conteo_clases / len(df) * 100
})
display(distribucion_clases)
''')
codigo('''# Verificar formalmente los requisitos con valores observados.
identificadores = [c for c in df.columns if c.lower() == "id"]
predictoras = [c for c in df.columns if c not in [objetivo] + identificadores]
clase_minoritaria = conteo_clases.idxmin()
porcentaje_minoritario = float(distribucion_clases.loc[clase_minoritaria, "Porcentaje"])
requisitos = {
    "Más de 100.000 registros": len(df) > 100_000,
    "Objetivo binario 0/1": set(df[objetivo].dropna().unique()) == {0, 1},
    "Clase minoritaria entre 1% y 30%": 1 <= porcentaje_minoritario <= 30,
    "Al menos 8 predictores": len(predictoras) >= 8,
    "Objetivo sin nulos": not df[objetivo].isna().any()
}
display(pd.DataFrame(requisitos.items(), columns=["Requisito", "Cumple"]))
print("Predictores:", len(predictoras))
print("Clase minoritaria:", clase_minoritaria, "| Porcentaje:", porcentaje_minoritario)
assert all(requisitos.values()) and clase_minoritaria == 1
with open(RAIZ / "data" / "validacion_porto_seguro.json", "w") as archivo:
    json.dump({"shape": list(df.shape), "predictores": len(predictoras),
               "clase_minoritaria": int(clase_minoritaria),
               "porcentaje_minoritario": porcentaje_minoritario,
               "requisitos": requisitos}, archivo, ensure_ascii=False, indent=2)
''')
md('''# 2. Exploración inicial

Se inspeccionan los datos originales sin imputar, eliminar filas ni ajustar transformaciones. `_cat` indica categorías nominales y `_bin` indicadores binarios. La descripción de OpenML identifica como ordinales las variables enteras restantes sin esos sufijos. No se interpreta la distancia entre categorías como una medida continua.
''')
codigo('''# Clasificar las columnas a partir del esquema documentado.
categoricas = [c for c in predictoras if c.endswith("_cat")]
binarias = [c for c in predictoras if c.endswith("_bin")]
ordinales = [
    "ps_ind_01", "ps_ind_03", "ps_ind_14", "ps_ind_15", "ps_car_11",
    "ps_calc_04", "ps_calc_05", "ps_calc_06", "ps_calc_07", "ps_calc_08",
    "ps_calc_09", "ps_calc_10", "ps_calc_11", "ps_calc_12", "ps_calc_13", "ps_calc_14"
]
numericas = [c for c in predictoras if c not in categoricas + binarias + ordinales]
grupos = {"Numérica": numericas, "Binaria": binarias,
          "Categórica": categoricas, "Ordinal": ordinales}
clasificacion_variables = pd.DataFrame([
    {"Variable": c, "Tipo": tipo, "dtype": str(df[c].dtype)}
    for tipo, columnas in grupos.items() for c in columnas
])
display(clasificacion_variables)
print("Identificadores excluidos:", identificadores)
print("Cantidad por tipo:", {tipo: len(cols) for tipo, cols in grupos.items()})
assert set(numericas + binarias + categoricas + ordinales) == set(predictoras)
''')
codigo('''# Revisar nulos, el código especial -1 y duplicados, sin reemplazar valores.
faltantes = pd.DataFrame({
    "Nulos": df[predictoras].isna().sum(),
    "Porcentaje_nulos": df[predictoras].isna().mean() * 100,
    "Codigo_menos_1": df[predictoras].eq(-1).sum()
})
display(faltantes)
print("Registros duplicados:", df.duplicated().sum())
display(df[numericas + ordinales].describe().T)
display(df[categoricas].nunique().to_frame("Cardinalidad_sin_nulos"))
''')
codigo('''# Contar posibles outliers solo en numéricas, sin eliminarlos.
filas_outliers = []
for columna in numericas:
    q1, q3 = df[columna].quantile([0.25, 0.75])
    iqr = q3 - q1
    limite_inferior, limite_superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    cantidad = ((df[columna] < limite_inferior) | (df[columna] > limite_superior)).sum()
    filas_outliers.append({"Variable": columna, "IQR": iqr,
                           "Posibles_outliers": int(cantidad)})
display(pd.DataFrame(filas_outliers))
''')
md('''`-1` representa faltantes en el concurso original; esta copia de OpenML puede traerlos como NaN. Se comprueban ambos. La conversión se hará dentro del pipeline; los ceros se conservan como valores válidos. La regla IQR es exploratoria y no justifica eliminar registros automáticamente. La revisión inicial no calcula estadísticas que se reutilicen para imputar ni selecciona variables según su relación con el objetivo.
''')
md('''# 3. Partición Train/Test

Se usa estratificación para conservar aproximadamente el desbalance en ambos conjuntos. Test queda reservado: no participa en tuning ni en la selección del ganador. No se ajusta ningún preprocesamiento antes de dividir.
''')
codigo('''from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV

X = df[predictoras].copy()
y = df[objetivo].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEMILLA
)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Porcentaje positivo train:", y_train.eq(1).mean() * 100)
print("Porcentaje positivo test:", y_test.eq(1).mean() * 100)
assert X_train.index.intersection(X_test.index).empty
''')
md('''# 4. Preprocesamiento

Todo el ajuste vive dentro del pipeline: conversión de `-1` a NaN, mediana para numéricas y moda para ordinales, binarias y categóricas. Las ordinales ya tienen códigos numéricos ordenados. Las nominales usan OneHotEncoder con categorías desconocidas ignoradas. No se escalan variables porque los modelos obligatorios son árboles o ensembles de árboles.
''')
codigo('''from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

def convertir_faltantes(datos):
    return datos.replace(-1, np.nan)

pipeline_numerico = Pipeline([
    ("imputar", SimpleImputer(strategy="median", keep_empty_features=True))
])
pipeline_discreto = Pipeline([
    ("imputar", SimpleImputer(strategy="most_frequent", keep_empty_features=True))
])
pipeline_categorico = Pipeline([
    ("imputar", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
    ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])
columnas_transformadas = ColumnTransformer([
    ("numericas", pipeline_numerico, numericas),
    ("ordinales_binarias", pipeline_discreto, ordinales + binarias),
    ("categoricas", pipeline_categorico, categoricas)
], remainder="drop", sparse_threshold=1.0)
preprocesador = Pipeline([
    ("faltantes", FunctionTransformer(convertir_faltantes, validate=False)),
    ("columnas", columnas_transformadas)
])
print(preprocesador)
''')
md('''En CV se clona y ajusta la preparación solamente dentro del entrenamiento de cada fold, evitando leakage. La salida dispersa limita el uso de memoria. SMOTE estándar, exigido en la Etapa A, puede interpolar códigos one-hot y generar dummy fraccionarias; se conserva el método exigido sin añadir otras técnicas.
''')
md('''# 5. Funciones auxiliares

Se reutilizan funciones pequeñas para métricas, costo, pipelines y entrenamiento. F1, recall y precisión se refieren siempre a la clase positiva 1. El tiempo se expresa en segundos.
''')
codigo('''from sklearn.metrics import (f1_score, recall_score, precision_score, roc_auc_score,
                             confusion_matrix, make_scorer)
from imblearn.pipeline import Pipeline as PipelineImblearn
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

scoring_f1 = make_scorer(f1_score, pos_label=1, zero_division=0)

def calcular_costo(y_real, pred, C_FP=100, C_FN=1000, pos_label=1):
    y_real, pred = np.asarray(y_real), np.asarray(pred)
    fp = int(((y_real != pos_label) & (pred == pos_label)).sum())
    fn = int(((y_real == pos_label) & (pred != pos_label)).sum())
    return fp * C_FP + fn * C_FN

def evaluar_clasificacion(y_real, pred, probabilidades=None, C_FP=100, C_FN=1000):
    tn, fp, fn, tp = confusion_matrix(y_real, pred, labels=[0, 1]).ravel()
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Precision": precision_score(y_real, pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_real, pred, pos_label=1, zero_division=0),
        "F1": f1_score(y_real, pred, pos_label=1, zero_division=0),
        "AUC_ROC": roc_auc_score(y_real, probabilidades) if probabilidades is not None else np.nan,
        "Costo_total": calcular_costo(y_real, pred, C_FP, C_FN)
    }
''')
codigo('''# Preparar un pipeline nuevo en cada ajuste, sin compartir estadísticas.
# Imblearn no admite un Pipeline de sklearn como paso intermedio: se despliega la preparación.
def crear_pipeline(estimador, estrategia):
    pasos = [(nombre, clone(paso)) for nombre, paso in preprocesador.steps]
    if estrategia == "SMOTE":
        pasos.append(("smote", SMOTE(random_state=SEMILLA)))
    pasos.append(("modelo", clone(estimador)))
    if estrategia == "SMOTE":
        return PipelineImblearn(pasos)
    return Pipeline(pasos)
''')
codigo('''# Buscar en la submuestra y ajustar después con TODO train.
def ajustar_modelo(nombre, estrategia, estimador, espacio):
    carpeta = RESULTADOS / estrategia.lower()
    carpeta.mkdir(exist_ok=True)
    busqueda = RandomizedSearchCV(
        crear_pipeline(estimador, estrategia), espacio, n_iter=10,
        scoring=scoring_f1, cv=cv, random_state=SEMILLA,
        n_jobs=-1, pre_dispatch="n_jobs", refit=False,
        return_train_score=True, error_score=np.nan
    )
    inicio = perf_counter()
    busqueda.fit(X_busqueda, y_busqueda)
    segundos = perf_counter() - inicio
    assert np.isfinite(busqueda.best_score_), "Ninguna configuración tuvo CV válida."
    i = busqueda.best_index_
    ganador = crear_pipeline(estimador, estrategia).set_params(**busqueda.best_params_)
    inicio_refit = perf_counter()
    ganador.fit(X_train, y_train)
    resultado = {
        "mejores_hiperparametros": busqueda.best_params_,
        "F1_CV": float(busqueda.best_score_),
        "F1_folds": [float(busqueda.cv_results_[f"split{k}_test_score"][i]) for k in range(3)],
        "F1_std": float(busqueda.cv_results_["std_test_score"][i]),
        "F1_train_folds": float(busqueda.cv_results_["mean_train_score"][i]),
        "Tiempo_busqueda": segundos,
        "Tiempo_refit": perf_counter() - inicio_refit,
        "Configuraciones_validas": int(np.isfinite(busqueda.cv_results_["mean_test_score"]).sum()),
        "Registros_refit": len(X_train)
    }
    pd.DataFrame(busqueda.cv_results_).to_csv(carpeta / f"{nombre}_cv.csv", index=False)
    with gzip.open(carpeta / f"{nombre}.pkl.gz", "wb", compresslevel=1) as archivo:
        cloudpickle.dump(ganador, archivo)
    print(nombre, estrategia, "F1 CV:", resultado["F1_CV"],
          "| configuraciones válidas:", resultado["Configuraciones_validas"], flush=True)
    return ganador, resultado
''')
codigo('''def crear_tabla_resultados(resultados_smote, resultados_pesos):
    filas = []
    for estrategia, resultados in [("SMOTE", resultados_smote), ("Pesos", resultados_pesos)]:
        for nombre, r in resultados.items():
            filas.append({
                "Estrategia": estrategia, "Modelo": nombre, "F1_CV": r["F1_CV"],
                "F1_fold_1": r["F1_folds"][0], "F1_fold_2": r["F1_folds"][1],
                "F1_fold_3": r["F1_folds"][2], "F1_std": r["F1_std"],
                "Tiempo_busqueda": r["Tiempo_busqueda"],
                "Observacion": r.get("Observacion", "")
            })
    return pd.DataFrame(filas).sort_values("F1_CV", ascending=False).reset_index(drop=True)
''')
md('''# 6. Etapa A — 5 modelos con SMOTE

Se busca sobre 30.000 filas estratificadas obtenidas solo de train para reducir tiempo de cómputo. Se conservan tres folds estratificados con semilla 2026 y F1 de la clase positiva como métrica. SMOTE está después de la preparación y dentro del pipeline: solo actúa durante el ajuste, nunca sobre validación o test. No se usan pesos en esta etapa.

Se paraleliza la búsqueda; los estimadores compatibles usan un hilo para evitar paralelismo anidado. Una vez elegida la configuración, se reajusta todo el pipeline con train completo.
''')
codigo('''X_busqueda, _, y_busqueda, _ = train_test_split(
    X_train, y_train, train_size=30_000, stratify=y_train, random_state=SEMILLA
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMILLA)
print("Submuestra:", X_busqueda.shape)
display(pd.DataFrame({"Train_%": y_train.value_counts(normalize=True) * 100,
                      "Submuestra_%": y_busqueda.value_counts(normalize=True) * 100}))
pd.Series(X_busqueda.index, name="indice_original").to_csv(
    RESULTADOS / "indices_submuestra.csv", index=False
)
modelos_base = {
    "DecisionTree": DecisionTreeClassifier(random_state=SEMILLA),
    "RandomForest": RandomForestClassifier(random_state=SEMILLA, n_jobs=1),
    "XGBoost": XGBClassifier(random_state=SEMILLA, n_jobs=1, tree_method="hist", eval_metric="logloss"),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=SEMILLA), random_state=SEMILLA),
    "LightGBM": LGBMClassifier(random_state=SEMILLA, n_jobs=1, verbosity=-1,
                               deterministic=True, force_col_wise=True)
}
''')
codigo('''# Espacios pequeños: al menos dos hiperparámetros por modelo.
espacios = {
    "DecisionTree": {"modelo__max_depth": [3, 5, 8, 12],
                     "modelo__min_samples_split": [10, 50, 100],
                     "modelo__min_samples_leaf": [5, 20, 50]},
    "RandomForest": {"modelo__n_estimators": [50, 75, 100],
                     "modelo__max_depth": [4, 6, 10], "modelo__min_samples_leaf": [10, 25, 50],
                     "modelo__max_features": ["sqrt", 0.5]},
    "XGBoost": {"modelo__n_estimators": [50, 100, 150], "modelo__max_depth": [2, 3, 5],
                "modelo__learning_rate": [0.03, 0.1, 0.2], "modelo__subsample": [0.7, 1.0]},
    "AdaBoost": {"modelo__n_estimators": [30, 50, 75],
                 "modelo__learning_rate": [0.05, 0.1, 0.5, 1.0],
                 "modelo__estimator__max_depth": [1, 2]},
    "LightGBM": {"modelo__n_estimators": [50, 100, 150], "modelo__num_leaves": [7, 15, 31],
                 "modelo__max_depth": [3, 5, -1], "modelo__learning_rate": [0.03, 0.1, 0.2]}
}
modelos_smote, resultados_smote = {}, {}
''')
for nombre in ['DecisionTree','RandomForest','XGBoost','AdaBoost','LightGBM']:
    codigo(f'''# Buscar y reajustar {nombre} con SMOTE.
modelos_smote["{nombre}"], resultados_smote["{nombre}"] = ajustar_modelo(
    "{nombre}", "SMOTE", modelos_base["{nombre}"], espacios["{nombre}"]
)
with open(RESULTADOS / "resultados_smote.json", "w") as archivo:
    json.dump(resultados_smote, archivo, ensure_ascii=False, indent=2)
''')
md('''# 7. Etapa B — 5 modelos con pesos

Se usa exactamente la misma submuestra y CV. No se incluye SMOTE. Decision Tree, Random Forest y LightGBM usan `class_weight="balanced"`; XGBoost usa la razón negativos/positivos calculada solo con `y_train`. AdaBoost tiene un árbol base con pesos balanced.

AdaBoost mantiene `DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=2026)`, sin SMOTE. Solo se buscan `n_estimators` y `learning_rate`. Con esta configuración el weak learner puede ser rechazado como peor que azar en un fold. Si ninguna configuración completa los tres folds, la combinación se registra como no válida, sin sustituir el estimador ni reutilizar resultados anteriores; queda excluida de la selección por CV. No se añaden pesos externos por `sample_weight`. LightGBM usa solo class_weight, sin is_unbalance ni scale_pos_weight.
''')
codigo('''modelos_con_pesos = {nombre: clone(modelo) for nombre, modelo in modelos_base.items()}
modelos_con_pesos["DecisionTree"].set_params(class_weight="balanced")
modelos_con_pesos["RandomForest"].set_params(class_weight="balanced")
scale_pos_weight = int(y_train.eq(0).sum()) / int(y_train.eq(1).sum())
modelos_con_pesos["XGBoost"].set_params(scale_pos_weight=scale_pos_weight)
modelos_con_pesos["AdaBoost"].set_params(
    estimator=DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=2026)
)
modelos_con_pesos["LightGBM"].set_params(class_weight="balanced")
espacios_pesos = {nombre: espacio.copy() for nombre, espacio in espacios.items()}
espacios_pesos["AdaBoost"] = {
    "modelo__n_estimators": [30, 50, 75],
    "modelo__learning_rate": [0.05, 0.1, 0.5, 1.0]
}
print("scale_pos_weight calculado solo con y_train:", scale_pos_weight)
modelos_pesos, resultados_pesos = {}, {}
''')
for nombre in ['DecisionTree','RandomForest','XGBoost','AdaBoost','LightGBM']:
    if nombre == 'AdaBoost':
        codigo('# Evaluar la configuración exigida; no reemplazar los ajustes fallidos.\nimport warnings\n\nbusqueda_adaboost = RandomizedSearchCV(\n    crear_pipeline(modelos_con_pesos["AdaBoost"], "Pesos"),\n    espacios_pesos["AdaBoost"], n_iter=10, scoring=scoring_f1,\n    cv=cv, random_state=SEMILLA, n_jobs=1, refit=False,\n    return_train_score=True, error_score=np.nan\n)\ninicio = perf_counter()\nwith warnings.catch_warnings(record=True) as avisos_adaboost:\n    warnings.simplefilter("always")\n    busqueda_adaboost.fit(X_busqueda, y_busqueda)\ntiempo_adaboost = perf_counter() - inicio\ncv_adaboost = pd.DataFrame(busqueda_adaboost.cv_results_)\ncv_adaboost.to_csv(RESULTADOS / "pesos/AdaBoost_cv.csv", index=False)\nvalidas_adaboost = np.isfinite(cv_adaboost["mean_test_score"])\nfallos_adaboost = int(cv_adaboost[[f"split{k}_test_score" for k in range(3)]].isna().sum().sum())\nmensajes_adaboost = [str(aviso.message) for aviso in avisos_adaboost]\n# Los avisos esperados quedan guardados y se resumen explícitamente.\nwith open(RESULTADOS / "pesos/AdaBoost_diagnostico.json", "w") as archivo:\n    json.dump({"registros_busqueda": len(X_busqueda), "folds": 3,\n               "configuraciones": len(cv_adaboost), "fits_fallidos": fallos_adaboost,\n               "avisos": mensajes_adaboost}, archivo, ensure_ascii=False, indent=2)\nif not validas_adaboost.any():\n    assert any("worse than random" in mensaje for mensaje in mensajes_adaboost), mensajes_adaboost\n    observacion = ("El weak learner obtuvo rendimiento peor que azar en al menos un fold "\n                   "y AdaBoost no pudo ajustarse con la configuración exigida.")\n    modelos_pesos["AdaBoost"] = None\n    resultados_pesos["AdaBoost"] = {\n        "mejores_hiperparametros": None, "F1_CV": None,\n        "F1_folds": [None, None, None], "F1_std": None, "F1_train_folds": None,\n        "Tiempo_busqueda": tiempo_adaboost, "Tiempo_refit": None,\n        "Configuraciones_validas": 0, "Registros_refit": 0,\n        "CV_valido": False, "Observacion": observacion\n    }\n    # Retirar el artefacto antiguo: no corresponde a la configuración exigida.\n    (RESULTADOS / "pesos/AdaBoost.pkl.gz").unlink(missing_ok=True)\n    print("AdaBoost + Pesos: No válido. " + observacion)\nelse:\n    i = cv_adaboost.loc[validas_adaboost, "mean_test_score"].idxmax()\n    mejores = cv_adaboost.loc[i, "params"]\n    modelo = crear_pipeline(modelos_con_pesos["AdaBoost"], "Pesos").set_params(**mejores)\n    inicio_refit = perf_counter()\n    modelo.fit(X_train, y_train)\n    modelos_pesos["AdaBoost"] = modelo\n    resultados_pesos["AdaBoost"] = {\n        "mejores_hiperparametros": mejores,\n        "F1_CV": float(cv_adaboost.loc[i, "mean_test_score"]),\n        "F1_folds": [float(cv_adaboost.loc[i, f"split{k}_test_score"]) for k in range(3)],\n        "F1_std": float(cv_adaboost.loc[i, "std_test_score"]),\n        "F1_train_folds": float(cv_adaboost.loc[i, "mean_train_score"]),\n        "Tiempo_busqueda": tiempo_adaboost, "Tiempo_refit": perf_counter() - inicio_refit,\n        "Configuraciones_validas": int(validas_adaboost.sum()), "Registros_refit": len(X_train),\n        "CV_valido": True, "Observacion": ""\n    }\n    with gzip.open(RESULTADOS / "pesos/AdaBoost.pkl.gz", "wb", compresslevel=1) as archivo:\n        cloudpickle.dump(modelo, archivo)\nprint("Fits fallidos:", fallos_adaboost, "/", len(cv_adaboost) * 3,\n      "| avisos registrados:", len(mensajes_adaboost))\nwith open(RESULTADOS / "resultados_pesos.json", "w") as archivo:\n    json.dump(resultados_pesos, archivo, ensure_ascii=False, indent=2, allow_nan=False)\n')
        continue
    codigo(f'''# Buscar y reajustar {nombre} con pesos.
modelos_pesos["{nombre}"], resultados_pesos["{nombre}"] = ajustar_modelo(
    "{nombre}", "Pesos", modelos_con_pesos["{nombre}"], espacios_pesos["{nombre}"]
)
with open(RESULTADOS / "resultados_pesos.json", "w") as archivo:
    json.dump(resultados_pesos, archivo, ensure_ascii=False, indent=2)
''')
md('''# 8. Comparación de las 10 combinaciones

El ganador se selecciona primero por F1 promedio de CV. Se revisan su dispersión y la diferencia entre F1 de ajuste y validación de los folds para describir estabilidad y posible sobreajuste. Estos scores provienen solo de la submuestra de train. No se ha evaluado test ni se afirma superioridad estadística con tres folds.
''')
codigo('''tabla_seleccion = crear_tabla_resultados(resultados_smote, resultados_pesos)
display(tabla_seleccion)
assert len(tabla_seleccion) == 10
combinaciones_validas = tabla_seleccion.dropna(subset=["F1_CV"])
print("Combinaciones con CV válido:", len(combinaciones_validas), "de 10")
ganador_cv = combinaciones_validas.iloc[0]
modelo_ganador_nombre = ganador_cv["Modelo"]
estrategia_ganadora = ganador_cv["Estrategia"]
modelos_estrategia = {"SMOTE": modelos_smote, "Pesos": modelos_pesos}
resultados_estrategia = {"SMOTE": resultados_smote, "Pesos": resultados_pesos}
modelo_ganador = modelos_estrategia[estrategia_ganadora][modelo_ganador_nombre]
resultado_ganador = resultados_estrategia[estrategia_ganadora][modelo_ganador_nombre]
print("Ganador elegido solo por CV:", modelo_ganador_nombre, estrategia_ganadora)
print("F1 CV:", resultado_ganador["F1_CV"], "| std:", resultado_ganador["F1_std"])
print("F1 de ajuste en folds:", resultado_ganador["F1_train_folds"])
print("Brecha ajuste-validación de folds:", resultado_ganador["F1_train_folds"] - resultado_ganador["F1_CV"])
tabla_seleccion.to_csv(RESULTADOS / "tabla_seleccion_cv.csv", index=False)
seleccion = {"Modelo": modelo_ganador_nombre, "Estrategia": estrategia_ganadora,
             "F1_CV": resultado_ganador["F1_CV"], "F1_std": resultado_ganador["F1_std"],
             "hiperparametros": resultado_ganador["mejores_hiperparametros"], "Test_usado": False}
with open(RESULTADOS / "ganador_cv.json", "w") as archivo:
    json.dump(seleccion, archivo, ensure_ascii=False, indent=2)
''')
codigo('''# Un solo gráfico de comparación por estrategia.
datos_grafico = tabla_seleccion.pivot(index="Modelo", columns="Estrategia", values="F1_CV")
ax = datos_grafico.plot.bar(figsize=(9, 4), rot=0)
ax.set_ylabel("F1 promedio de CV")
ax.set_title("Cinco modelos y dos estrategias (NaN: CV no válido)")
ax.legend(title="Estrategia", loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.savefig(RESULTADOS / "comparacion_cv.png", dpi=150)
plt.show()
''')
codigo('''# Revisar las mejores alternativas, sin consultar test.
revision_cv = tabla_seleccion.head(3).copy()
revision_cv["F1_ajuste_folds"] = [
    resultados_estrategia[fila.Estrategia][fila.Modelo]["F1_train_folds"]
    for fila in revision_cv.itertuples()
]
revision_cv["Brecha_ajuste_CV"] = revision_cv["F1_ajuste_folds"] - revision_cv["F1_CV"]
display(revision_cv)
display(Markdown(
    f"Se selecciona **{modelo_ganador_nombre} + {estrategia_ganadora}**, con F1 CV "
    f"**{resultado_ganador['F1_CV']:.5f}** y std **{resultado_ganador['F1_std']:.5f}**. "
    "La tabla permite revisar estabilidad y sobreajuste aparente; no prueba que el modelo "
    "esté libre de sobreajuste. Test permanece reservado para reportar resultados."
))
''')
md('''# 9. Tabla final obligatoria de las 10 combinaciones

La selección anterior ya está cerrada. Ahora se consulta test solo para reportar métricas de los diez pipelines finales. F1 Train se calcula con las observaciones originales, no con las filas sintéticas de SMOTE. Se usa predict normal; los umbrales de costos se tratarán después. Test no cambia el ganador ni se usa para elegir posteriormente los umbrales.
''')
codigo('''filas_finales = []
for estrategia, modelos in modelos_estrategia.items():
    for nombre, pipeline in modelos.items():
        r = resultados_estrategia[estrategia][nombre]
        if pipeline is None:
            filas_finales.append({
                "Estrategia": estrategia, "Modelo": nombre,
                **{col: np.nan for col in ["F1_Train", "F1_CV", "F1_Test", "Brecha_Train_Test",
                                          "Recall_Test", "Precision_Test", "AUC_ROC_Test"]},
                "Tiempo_busqueda": r["Tiempo_busqueda"], "Ganador_CV": False,
                "Observacion": r["Observacion"]
            })
            continue
        # La actualización puntual reutiliza métricas guardadas de modelos sin cambios.
        if "metricas_finales_guardadas" in globals():
            fila = metricas_finales_guardadas.loc[
                (metricas_finales_guardadas.Estrategia == estrategia) &
                (metricas_finales_guardadas.Modelo == nombre)
            ].iloc[0].to_dict()
            fila["Observacion"] = r.get("Observacion", "")
            filas_finales.append(fila)
            continue
        pred_train = pipeline.predict(X_train)
        pred_test = pipeline.predict(X_test)
        proba_test = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(1)]
        metricas = evaluar_clasificacion(y_test, pred_test, proba_test)
        f1_train = f1_score(y_train, pred_train, pos_label=1, zero_division=0)
        r = resultados_estrategia[estrategia][nombre]
        filas_finales.append({
            "Estrategia": estrategia, "Modelo": nombre, "F1_Train": f1_train,
            "F1_CV": r["F1_CV"], "F1_Test": metricas["F1"],
            "Brecha_Train_Test": f1_train - metricas["F1"],
            "Recall_Test": metricas["Recall"], "Precision_Test": metricas["Precision"],
            "AUC_ROC_Test": metricas["AUC_ROC"], "Tiempo_busqueda": r["Tiempo_busqueda"],
            "Ganador_CV": nombre == modelo_ganador_nombre and estrategia == estrategia_ganadora,
            "Observacion": r.get("Observacion", "")
        })
tabla_10_combinaciones = pd.DataFrame(filas_finales).sort_values("F1_CV", ascending=False).reset_index(drop=True)
display(tabla_10_combinaciones)
tabla_10_combinaciones.to_csv(RESULTADOS / "tabla_10_combinaciones.csv", index=False)
fila_ganador_final = tabla_10_combinaciones.loc[tabla_10_combinaciones["Ganador_CV"]].iloc[0]
assert tabla_10_combinaciones["Ganador_CV"].sum() == 1
print("El ganador por CV se conserva:", modelo_ganador_nombre, estrategia_ganadora)
''')
md('''# 10. Costos y probabilidades reales

Se crea un nuevo pipeline con el mismo tipo de modelo ganador y sus hiperparámetros, quitando SMOTE y todos los pesos de desbalance. No se vuelve a buscar hiperparámetros. Se conservan las proporciones reales de train; esto no garantiza calibración de probabilidades.

Los costos son **supuestos académicos**, no costos reales de Porto Seguro. FP: considerar riesgoso a quien no presenta siniestro, costo 100. FN: considerar de bajo riesgo a quien sí presenta siniestro, costo 1000. Se supone que FN cuesta diez veces FP y que los aciertos tienen costo cero.
''')
codigo('''C_FP, C_FN = 100, 1000
estimador_probabilidades = clone(modelo_ganador.named_steps["modelo"])
parametros_estimador = estimador_probabilidades.get_params(deep=True)
# Retirar pesos también del árbol base si el ganador fuera AdaBoost.
ajustes_sin_pesos = {p: None for p in parametros_estimador if p.endswith("class_weight")}
if "scale_pos_weight" in parametros_estimador:
    ajustes_sin_pesos["scale_pos_weight"] = None
if "is_unbalance" in parametros_estimador:
    ajustes_sin_pesos["is_unbalance"] = False
estimador_probabilidades.set_params(**ajustes_sin_pesos)
modelo_probabilidades = crear_pipeline(estimador_probabilidades, "Sin balanceo")
modelo_probabilidades.fit(X_train, y_train)
probabilidades_train = modelo_probabilidades.predict_proba(X_train)[:, list(modelo_probabilidades.classes_).index(1)]
with gzip.open(RESULTADOS / "modelo_probabilidades.pkl.gz", "wb", compresslevel=1) as archivo:
    cloudpickle.dump(modelo_probabilidades, archivo)
print("Modelo sin balanceo:", type(estimador_probabilidades).__name__)
print("Proporción positiva real:", y_train.eq(1).mean())
''')
md('''# 11. Umbral teórico y empírico

Con costo cero para aciertos y probabilidades que reflejen el riesgo real, el umbral teórico es `C_FP / (C_FP + C_FN)`. El manual minimiza el costo sobre las filas originales de train mediante una grilla de 0,01 a 0,50. Este mínimo es dentro de muestra y puede ser optimista. Test no interviene.
''')
codigo('''p_teorico = C_FP / (C_FP + C_FN)
grilla_umbrales = np.arange(1, 51) / 100
filas_costos = []
for umbral in grilla_umbrales:
    pred = (probabilidades_train >= umbral).astype(int)
    matriz = confusion_matrix(y_train, pred, labels=[0, 1])
    filas_costos.append({"Umbral": float(umbral), "FP": int(matriz[0, 1]),
                        "FN": int(matriz[1, 0]),
                        "Costo_train": calcular_costo(y_train, pred, C_FP, C_FN)})
tabla_costos_train = pd.DataFrame(filas_costos)
umbral_empirico_manual = float(tabla_costos_train.loc[tabla_costos_train["Costo_train"].idxmin(), "Umbral"])
display(tabla_costos_train)
print("Teórico:", p_teorico, "| Manual:", umbral_empirico_manual)
print("Costo mínimo en train:", tabla_costos_train["Costo_train"].min())
tabla_costos_train.to_csv(RESULTADOS / "costos_umbrales_train.csv", index=False)
''')
codigo('''fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(tabla_costos_train["Umbral"], tabla_costos_train["Costo_train"])
for nombre, umbral, color in [("0.5", 0.5, "gray"), ("Teórico", p_teorico, "blue"),
                              ("Manual", umbral_empirico_manual, "orange")]:
    ax.axvline(umbral, linestyle="--", color=color, label=f"{nombre}: {umbral:.4f}")
ax.set_xlabel("Umbral")
ax.set_ylabel("Costo en train (unidades monetarias)")
ax.set_title("Costo dentro de muestra por umbral")
ax.set_xlim(0, 0.52)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.savefig(RESULTADOS / "costo_umbral_train.png", dpi=150)
plt.show()
''')
md('''# 12. TunedThresholdClassifierCV

Se minimiza el mismo costo con signo negativo mediante tres folds estratificados de train y la misma grilla. La CV ajusta copias del modelo con hiperparámetros fijos; no es una nueva búsqueda de modelos. Se reajusta su copia final con todo train. No se usa cv=prefit sobre los mismos datos de ajuste ni se consulta test para elegir umbral.
''')
codigo('''try:
    from sklearn.model_selection import TunedThresholdClassifierCV
except ImportError as error:
    raise ImportError('Se requiere scikit-learn >= 1.5. Instala: python -m pip install "scikit-learn>=1.5"') from error

scorer_costo = make_scorer(calcular_costo, greater_is_better=False,
                          C_FP=C_FP, C_FN=C_FN, pos_label=1)
ajustador_umbral = TunedThresholdClassifierCV(
    clone(modelo_probabilidades), scoring=scorer_costo, response_method="predict_proba",
    thresholds=grilla_umbrales,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMILLA),
    refit=True, n_jobs=1, random_state=SEMILLA, store_cv_results=True
)
ajustador_umbral.fit(X_train, y_train)
umbral_tuned = float(ajustador_umbral.best_threshold_)
tabla_umbrales = pd.DataFrame({
    "Metodo": ["0.5", "Teórico", "Manual (train)", "Tuned (CV de train)"],
    "Umbral": [0.5, p_teorico, umbral_empirico_manual, umbral_tuned]
})
display(tabla_umbrales)
tabla_umbrales.to_csv(RESULTADOS / "umbrales.csv", index=False)
with gzip.open(RESULTADOS / "ajustador_umbral.pkl.gz", "wb", compresslevel=1) as archivo:
    cloudpickle.dump(ajustador_umbral, archivo)
''')
md('''# 13. Evaluación final en Test

Se aplican las políticas predefinidas al modelo de probabilidades sin balanceo. El ganador seleccionado por CV también se reporta con su predict normal. Se conservan todos los umbrales fijados en train: ningún resultado de test modifica modelo o umbral. El ahorro se calcula contra la política de umbral 0,5, con los mismos costos académicos.
''')
codigo('''probabilidades_test = modelo_probabilidades.predict_proba(X_test)[:, list(modelo_probabilidades.classes_).index(1)]
politicas = {
    "Todo 0": np.zeros(len(y_test), dtype=int),
    "Umbral 0.5": (probabilidades_test >= 0.5).astype(int),
    "Teórico": (probabilidades_test >= p_teorico).astype(int),
    "Manual": (probabilidades_test >= umbral_empirico_manual).astype(int),
    "Tuned": (probabilidades_test >= umbral_tuned).astype(int),
    f"Ganador CV: {modelo_ganador_nombre} + {estrategia_ganadora}": modelo_ganador.predict(X_test)
}
filas_politicas = []
for politica, pred in politicas.items():
    m = evaluar_clasificacion(y_test, pred, C_FP=C_FP, C_FN=C_FN)
    filas_politicas.append({"Politica": politica, **m})
comparacion_test = pd.DataFrame(filas_politicas).drop(columns="AUC_ROC")
costo_0_5 = int(comparacion_test.loc[comparacion_test["Politica"] == "Umbral 0.5", "Costo_total"].iloc[0])
comparacion_test["Ahorro_vs_0.5"] = costo_0_5 - comparacion_test["Costo_total"]
comparacion_test = comparacion_test.sort_values("Costo_total", kind="stable").reset_index(drop=True)
display(comparacion_test)
comparacion_test.to_csv(RESULTADOS / "comparacion_test.csv", index=False)
''')
codigo('''ax = comparacion_test.plot.bar(x="Politica", y="Costo_total", figsize=(9, 4), legend=False, rot=15)
ax.set_ylabel("Costo en test (unidades monetarias)")
ax.set_title("Políticas predefinidas: costo final")
plt.tight_layout()
plt.savefig(RESULTADOS / "costos_politicas_test.png", dpi=150)
plt.show()
# Matrices como tablas para no agregar gráficos innecesarios.
for politica in ["Umbral 0.5", "Teórico", "Tuned"]:
    print(politica)
    display(pd.DataFrame(confusion_matrix(y_test, politicas[politica], labels=[0, 1]),
                         index=["Real 0", "Real 1"], columns=["Predicho 0", "Predicho 1"]))
''')
codigo('''menor_costo = comparacion_test.iloc[0]
display(Markdown(
    f"El menor costo observado corresponde a **{menor_costo['Politica']}**: "
    f"**{int(menor_costo['Costo_total']):,}**, con ahorro **{int(menor_costo['Ahorro_vs_0.5']):,}** "
    "frente a 0,5. Bajar el umbral permite más positivos, pero también puede aumentar FP. "
    "El mínimo manual de train es optimista; la evaluación de test no se usa para recalcularlo "
    "ni para cambiar el ganador por CV. Los valores son académicos y no implican causalidad."
))
''')
md('''# 14. Análisis de sensibilidad

Exactamente cuatro escenarios, con FP=100 y FN=200, 500, 1000 y 2000. Los conteos de la grilla de train se reutilizan porque no dependen de los costos. Cada umbral se elige solo con train y luego se aplica una vez a las probabilidades de test. No se selecciona un escenario con test.
''')
codigo('''filas_sensibilidad = []
for costo_fn in [200, 500, 1000, 2000]:
    costos_escenario = tabla_costos_train["FP"] * 100 + tabla_costos_train["FN"] * costo_fn
    umbral_escenario = float(tabla_costos_train.loc[costos_escenario.idxmin(), "Umbral"])
    pred = (probabilidades_test >= umbral_escenario).astype(int)
    m = evaluar_clasificacion(y_test, pred, C_FP=100, C_FN=costo_fn)
    filas_sensibilidad.append({
        "Escenario": len(filas_sensibilidad) + 1, "C_FP": 100, "C_FN": costo_fn,
        "Ratio_FN_FP": costo_fn / 100, "Umbral_teorico": 100 / (100 + costo_fn),
        "Umbral_empirico_train": umbral_escenario, "FP_test": m["FP"], "FN_test": m["FN"],
        "Precision_test": m["Precision"], "Recall_test": m["Recall"],
        "F1_test": m["F1"], "Costo_test": m["Costo_total"]
    })
sensibilidad_costos = pd.DataFrame(filas_sensibilidad)
display(sensibilidad_costos)
sensibilidad_costos.to_csv(RESULTADOS / "sensibilidad_costos.csv", index=False)
''')
codigo('''# Solo dos gráficos simples.
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(sensibilidad_costos["Ratio_FN_FP"], sensibilidad_costos["Umbral_empirico_train"], marker="o")
ax.set_xlabel("Ratio FN/FP")
ax.set_ylabel("Umbral empírico de train")
ax.set_xticks(sensibilidad_costos["Ratio_FN_FP"])
plt.tight_layout()
plt.savefig(RESULTADOS / "sensibilidad_umbral.png", dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(sensibilidad_costos["Ratio_FN_FP"], sensibilidad_costos["Costo_test"], marker="o")
ax.set_xlabel("Ratio FN/FP")
ax.set_ylabel("Costo total en test")
ax.set_xticks(sensibilidad_costos["Ratio_FN_FP"])
plt.tight_layout()
plt.savefig(RESULTADOS / "sensibilidad_costo.png", dpi=150)
plt.show()
''')
md('''Cuando FN cuesta relativamente más, el umbral tiende a bajar y el recall aumenta, generalmente a costa de más FP. Todos los umbrales de los escenarios se eligieron con train, nunca test. Los costos totales de escenarios distintos también cambian por el valor monetario de FN, no solo por la clasificación.
''')
md('''# 15. Preparación para el dashboard

Se exportan únicamente etiquetas y probabilidades del modelo sin balanceo original. Permiten recalcular decisiones, matriz de confusión, métricas y costos para cualquier umbral sin entrenar. La configuración contiene los costos y umbrales calculados. No se construye Streamlit.
''')
codigo('''carpeta_dashboard = RAIZ / "dashboard"
carpeta_dashboard.mkdir(exist_ok=True)
datos_dashboard = pd.DataFrame({"y_real": y_test.to_numpy(),
                                "probabilidad_clase_1": probabilidades_test})
datos_dashboard.to_csv(carpeta_dashboard / "datos_dashboard.csv", index=False)
config_dashboard = {"C_FP": C_FP, "C_FN": C_FN,
                    "umbral_teorico": p_teorico,
                    "umbral_empirico_manual": umbral_empirico_manual,
                    "umbral_tuned": umbral_tuned}
with open(carpeta_dashboard / "config.json", "w") as archivo:
    json.dump(config_dashboard, archivo, indent=2)
print("Filas exportadas:", len(datos_dashboard))
display(datos_dashboard.head())
''')
md('''# 16. Resumen final

Los siguientes valores se generan a partir de esta ejecución; no se fijan ganadores ni métricas de antemano.
''')
codigo('''estrategia_general = tabla_seleccion.groupby("Estrategia")["F1_CV"].mean().idxmax()
resumen_final = {
    "dataset": "Porto Seguro", "registros": len(df),
    "porcentaje_positivo": float(y.eq(1).mean() * 100),
    "combinaciones": len(tabla_10_combinaciones), "estrategia_general_por_media_CV": estrategia_general,
    "ganador": modelo_ganador_nombre, "estrategia_ganadora": estrategia_ganadora,
    "F1_CV": float(resultado_ganador["F1_CV"]), "F1_Test": float(fila_ganador_final["F1_Test"]),
    "C_FP": C_FP, "C_FN": C_FN, "umbral_teorico": p_teorico,
    "umbral_empirico_manual": umbral_empirico_manual, "umbral_tuned": umbral_tuned,
    "politica_menor_costo_test": menor_costo["Politica"],
    "costo_menor_politica_test": int(menor_costo["Costo_total"])
}
with open(RESULTADOS / "resumen_final.json", "w") as archivo:
    json.dump(resumen_final, archivo, ensure_ascii=False, indent=2)
texto_resumen = f"""**Porto Seguro:** {len(df):,} registros, clase positiva {y.eq(1).mean()*100:.2f}%.

**5 modelos × 2 estrategias.** Mejor estrategia por F1 CV medio: {estrategia_general}.
Ganador: **{modelo_ganador_nombre} + {estrategia_ganadora}**;
F1 CV **{resultado_ganador['F1_CV']:.5f}**, F1 Test **{fila_ganador_final['F1_Test']:.5f}**.

**Costos académicos:** FP={C_FP}, FN={C_FN}.
Umbral teórico **{p_teorico:.4f}**, manual **{umbral_empirico_manual:.2f}**, tuned **{umbral_tuned:.2f}**.
Menor costo final observado: **{menor_costo['Politica']}**, **{int(menor_costo['Costo_total']):,}**.

Mejor F1 no garantiza menor costo. Los costos son supuestos, las asociaciones no demuestran
causalidad y test no cambió la selección por CV ni los umbrales elegidos con train.
Los datos del futuro dashboard quedaron exportados sin construir una aplicación."""
display(Markdown(texto_resumen))
(RESULTADOS / "resumen_final.md").write_text(texto_resumen, encoding="utf-8")
''')
md('''**Cierre:** las métricas y costos corresponden a los modelos y supuestos de esta ejecución. No se afirma causalidad ni se presentan los costos como cifras reales de Porto Seguro.
''')

nb = nbformat.v4.new_notebook(cells=celdas, metadata={
    'kernelspec': {'name': 'certamen-local', 'display_name': 'Python (Certamen 1)', 'language': 'python'},
    'language_info': {'name': 'python'}
})
nbformat.validate(nb)
for celda in celdas:
    if celda.cell_type == 'code': compile(celda.source, 'notebook', 'exec')
ruta = RAIZ / 'notebooks' / 'Certamen1_Grupo_X.ipynb'
nbformat.write(nb, ruta)
print('Notebook creado:', ruta)
print('Secciones:', sum(c.cell_type == 'markdown' and c.source.startswith('# ') for c in celdas))
print('Celdas:', len(celdas))
