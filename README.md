# Certamen 1 — Machine Learning

Clasificación binaria con Porto Seguro (OpenML 42742), cinco modelos por dos estrategias de desbalance. El proyecto incluye selección por CV, evaluación final, costos, umbrales y un dashboard interactivo en Streamlit.

## Ejecutar

1. Desde esta carpeta, crea un entorno e instala dependencias:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```
2. En VS Code, abre `notebooks/Certamen1_Grupo_X.ipynb`, selecciona el intérprete de `.venv` y ejecuta **Run All**. También puedes ejecutar:
   ```bash
   python scripts/ejecutar_notebook.py
   ```
   Este script inicia un kernel nuevo y guarda los outputs después de cada celda. No establece límite de tiempo para las búsquedas.
3. Si `data/raw/porto_seguro.csv` no existe, el notebook lo descarga automáticamente. Para descargar nuevamente:
   ```bash
   python scripts/descargar_porto_seguro.py
   ```

En macOS, XGBoost y LightGBM necesitan el runtime OpenMP. Si el sistema informa que falta `libomp.dylib`, instala OpenMP con `brew install libomp`. En el entorno usado aquí el runtime está disponible mediante Anaconda.

## Dashboard

El dashboard está en `dashboard/app.py`. Usa `dashboard/datos_dashboard.csv` y `dashboard/config.json` para visualizar métricas, costos y umbrales; no reentrena modelos ni recalcula sus probabilidades.

Desde la raíz del proyecto, con el entorno activado:

```bash
streamlit run dashboard/app.py
```

## Archivos

- `data/raw/porto_seguro.csv`: copia original sin limpieza avanzada.
- `notebooks/Certamen1_Grupo_X.ipynb`: 16 secciones, con outputs reales.
- `resultados/`: búsquedas completas, parámetros, pipelines ajustados, tablas, gráficos y resumen.
- `resultados/control_ejecucion.json`: estado del último Run All.
- `dashboard/app.py`: dashboard Streamlit.
- `dashboard/datos_dashboard.csv`: etiquetas y probabilidades de test.
- `dashboard/config.json`: costos académicos y umbrales elegidos con train.
- `scripts/construir_notebook.py`: fuente para regenerar el notebook; regenerarlo reemplaza sus outputs. No es necesario para ejecutarlo.

La búsqueda usa 30.000 filas estratificadas solo de train, 10 combinaciones y 3 folds por modelo. Cada configuración ganadora con CV válido se ajusta luego con todo train. AdaBoost con pesos conserva el árbol base de profundidad 1 y se registra como no válido por fallos en CV; el ganador se selecciona entre las 9 combinaciones válidas. La selección usa F1 positivo de CV; test se usa solo para reportar. Los umbrales se eligen exclusivamente con train. El costo manual dentro de muestra puede ser optimista. Los costos FP/FN son supuestos académicos, no cifras reales de Porto Seguro.

`data/raw/` y `resultados/` están ignorados mediante `.gitignore` y no se incluyen en el repositorio de GitHub. Pueden regenerarse localmente descargando la base y ejecutando el notebook.
