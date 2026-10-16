"""Verificar archivos, ejecución completa y coherencia de resultados sin entrenar."""
from pathlib import Path
import json
import gzip
import cloudpickle
import numpy as np
import pandas as pd
import nbformat

RAIZ = Path(__file__).resolve().parents[1]
archivos = [
    'data/raw/porto_seguro.csv', 'notebooks/Certamen1_Grupo_X.ipynb',
    'dashboard/datos_dashboard.csv', 'dashboard/config.json', 'README.md', 'requirements.txt'
]
for nombre in archivos:
    ruta = RAIZ / nombre
    assert ruta.is_file() and ruta.stat().st_size > 0, nombre
    assert ruta.resolve().is_relative_to(RAIZ)
control = json.loads((RAIZ / 'resultados/control_ejecucion.json').read_text())
assert control['Run_All_sin_errores'] is True
nb = nbformat.read(RAIZ / archivos[1], as_version=4)
nbformat.validate(nb)
secciones = [c.source.splitlines()[0] for c in nb.cells if c.cell_type == 'markdown' and c.source.startswith('# ')]
assert len(secciones) == 16
for i, titulo in enumerate(secciones, start=1):
    assert titulo.startswith(f'# {i}.')
for celda in nb.cells:
    if celda.cell_type == 'code':
        assert celda.execution_count is not None
        assert not any(o.output_type == 'error' for o in celda.outputs)

seleccion = pd.read_csv(RAIZ / 'resultados/tabla_seleccion_cv.csv')
final = pd.read_csv(RAIZ / 'resultados/tabla_10_combinaciones.csv')
assert len(seleccion) == len(final) == 10
assert final.F1_CV.dropna().is_monotonic_decreasing and final.Ganador_CV.sum() == 1
resumen = json.loads((RAIZ / 'resultados/resumen_final.json').read_text())
assert resumen['ganador'] == seleccion.iloc[0].Modelo
assert resumen['estrategia_ganadora'] == seleccion.iloc[0].Estrategia
assert np.allclose(final.Brecha_Train_Test, final.F1_Train - final.F1_Test, equal_nan=True)
for estrategia in ['smote', 'pesos']:
    resultados = json.loads((RAIZ / f'resultados/resultados_{estrategia}.json').read_text())
    assert len(resultados) == 5
    for nombre, r in resultados.items():
        assert len(pd.read_csv(RAIZ / f'resultados/{estrategia}/{nombre}_cv.csv')) == 10
        if r.get('CV_valido') is False:
            assert estrategia == 'pesos' and nombre == 'AdaBoost'
            assert r['F1_CV'] is None and r['Configuraciones_validas'] == 0
            assert not (RAIZ / f'resultados/{estrategia}/{nombre}.pkl.gz').exists()
            fila = final.loc[(final.Estrategia == 'Pesos') & (final.Modelo == nombre)].iloc[0]
            assert pd.isna(fila.F1_CV) and pd.isna(fila.F1_Test) and bool(fila.Observacion)
            continue
        assert np.isclose(np.mean(r['F1_folds']), r['F1_CV'])
        assert np.isclose(np.std(r['F1_folds']), r['F1_std'])
        assert r['Configuraciones_validas'] > 0
        with gzip.open(RAIZ / f'resultados/{estrategia}/{nombre}.pkl.gz', 'rb') as archivo:
            pipeline = cloudpickle.load(archivo)
        assert list(pipeline.classes_) == [0, 1]
        assert ('smote' in pipeline.named_steps) == (estrategia == 'smote')
        assert pipeline.named_steps['modelo'].random_state == 2026

exportado = pd.read_csv(RAIZ / 'dashboard/datos_dashboard.csv')
assert exportado.columns.tolist() == ['y_real', 'probabilidad_clase_1']
assert set(exportado.y_real.unique()) == {0, 1}
assert exportado.probabilidad_clase_1.between(0, 1).all()
config = json.loads((RAIZ / 'dashboard/config.json').read_text())
assert np.isclose(config['umbral_teorico'], config['C_FP'] / (config['C_FP'] + config['C_FN']))
politicas = pd.read_csv(RAIZ / 'resultados/comparacion_test.csv')
assert len(politicas) == 6 and politicas.Costo_total.is_monotonic_increasing
for etiqueta, umbral in [('Umbral 0.5', 0.5), ('Teórico', config['umbral_teorico']),
                         ('Manual', config['umbral_empirico_manual']), ('Tuned', config['umbral_tuned'])]:
    pred = exportado.probabilidad_clase_1.to_numpy() >= umbral
    y = exportado.y_real.to_numpy()
    fp, fn = int(((y == 0) & pred).sum()), int(((y == 1) & ~pred).sum())
    fila = politicas.loc[politicas.Politica == etiqueta].iloc[0]
    assert fila.FP == fp and fila.FN == fn
    assert fila.Costo_total == fp * config['C_FP'] + fn * config['C_FN']
sensibilidad = pd.read_csv(RAIZ / 'resultados/sensibilidad_costos.csv')
assert sensibilidad.C_FN.tolist() == [200, 500, 1000, 2000]
assert sensibilidad.Ratio_FN_FP.tolist() == [2, 5, 10, 20]
assert sensibilidad.Umbral_empirico_train.is_monotonic_decreasing
verificacion = {'archivos_obligatorios': archivos, 'secciones': len(secciones),
                'notebook_JSON_valido': True, 'Run_All_sin_errores': True,
                'outputs_conservados': True, 'combinaciones_CV_validas': int(final.F1_CV.notna().sum()),
                'modelos_guardados': len(list((RAIZ / 'resultados/smote').glob('*.pkl.gz'))) +
                                     len(list((RAIZ / 'resultados/pesos').glob('*.pkl.gz'))),
                'dashboard_reproduce_politicas': True,
                'carpeta_proyecto': str(RAIZ)}
(RAIZ / 'resultados/verificacion_final.json').write_text(json.dumps(verificacion, ensure_ascii=False, indent=2))
print(json.dumps(resumen, ensure_ascii=False, indent=2))
print('VERIFICACIÓN FINAL COMPLETA: todos los controles pasaron.')
