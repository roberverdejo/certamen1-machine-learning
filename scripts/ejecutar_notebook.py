"""Ejecutar Run All real y guardar salidas, sin usar resultados anteriores."""
from pathlib import Path
import os
import sys
import json
from datetime import datetime

RAIZ = Path(__file__).resolve().parents[1]
os.chdir(RAIZ)
# Mantener archivos temporales y cachés dentro del proyecto.
for variable, carpeta in {
    'IPYTHONDIR': '.cache/ipython', 'MPLCONFIGDIR': '.cache/matplotlib',
    'XDG_CACHE_HOME': '.cache', 'JUPYTER_RUNTIME_DIR': '.cache/jupyter/runtime',
    'TMPDIR': '.cache/tmp', 'JOBLIB_TEMP_FOLDER': '.cache/joblib'
}.items():
    ruta = RAIZ / carpeta
    ruta.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(ruta)
os.environ['JUPYTER_PATH'] = str(RAIZ / '.cache/jupyter')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['SSL_CERT_FILE'] = '/etc/ssl/cert.pem' if Path('/etc/ssl/cert.pem').exists() else os.environ.get('SSL_CERT_FILE', '')
# Evitar multiplicar hilos nativos cuando RandomizedSearchCV paraleliza folds.
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '1'

import nbformat
from nbclient import NotebookClient

carpeta_kernel = RAIZ / '.cache/jupyter/kernels/certamen-local'
carpeta_kernel.mkdir(parents=True, exist_ok=True)
(carpeta_kernel / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Python (Certamen 1)', 'language': 'python'
}), encoding='utf-8')
ruta_notebook = RAIZ / 'notebooks/Certamen1_Grupo_X.ipynb'
notebook = nbformat.read(ruta_notebook, as_version=4)


def guardar_avance(cell, cell_index, **kwargs):
    nbformat.write(notebook, ruta_notebook)
    print(f'Celda {cell_index + 1}/{len(notebook.cells)}: {cell.source.splitlines()[0][:100]}', flush=True)
    for salida in cell.get('outputs', []):
        if salida.output_type == 'stream' and salida.name == 'stdout':
            print(salida.text[-1600:], flush=True)


inicio = datetime.now().isoformat()
control = {'inicio': inicio, 'Run_All_sin_errores': False}
ruta_control = RAIZ / 'resultados/control_ejecucion.json'
ruta_control.write_text(json.dumps(control, indent=2), encoding='utf-8')
cliente = NotebookClient(notebook, timeout=None, kernel_name='certamen-local',
                         resources={'metadata': {'path': str(RAIZ)}},
                         allow_errors=False, on_cell_executed=guardar_avance)
try:
    cliente.execute()
except Exception:
    nbformat.write(notebook, ruta_notebook)
    raise
nbformat.validate(notebook)
nbformat.write(notebook, ruta_notebook)
control.update({'fin': datetime.now().isoformat(), 'Run_All_sin_errores': True,
                'secciones': sum(c.cell_type == 'markdown' and c.source.startswith('# ') for c in notebook.cells),
                'celdas_codigo': sum(c.cell_type == 'code' for c in notebook.cells)})
ruta_control.write_text(json.dumps(control, ensure_ascii=False, indent=2), encoding='utf-8')
print('RUN ALL COMPLETO SIN ERRORES; outputs conservados.', flush=True)
