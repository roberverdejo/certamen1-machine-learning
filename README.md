# Certamen 1 — Machine Learning

Clasificación binaria con Porto Seguro (OpenML 42742), cinco modelos por dos estrategias de desbalance. El notebook incluye selección por CV, evaluación final, costos, umbrales y exportación de datos para un dashboard futuro.

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

## Archivos

- `data/raw/porto_seguro.csv`: copia original sin limpieza avanzada.
- `notebooks/Certamen1_Grupo_X.ipynb`: 16 secciones, con outputs reales.
- `resultados/`: búsquedas completas, parámetros, pipelines ajustados, tablas, gráficos y resumen.
- `resultados/control_ejecucion.json`: estado del último Run All.
- `dashboard/datos_dashboard.csv`: etiquetas y probabilidades de test.
- `dashboard/config.json`: costos académicos y umbrales elegidos con train.
- `scripts/construir_notebook.py`: fuente para regenerar el notebook; regenerarlo reemplaza sus outputs. No es necesario para ejecutarlo.

La búsqueda usa 30.000 filas estratificadas solo de train, 10 combinaciones y 3 folds por modelo. Cada ganador se ajusta luego con todo train. La selección usa F1 positivo de CV; test se usa solo para reportar. Los umbrales se eligen exclusivamente con train. El costo manual dentro de muestra puede ser optimista. Los costos FP/FN son supuestos académicos, no cifras reales de Porto Seguro.

No se incluye Streamlit, presentación, PDF ni guion de exposición.
