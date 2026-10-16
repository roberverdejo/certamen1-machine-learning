"""Descargar la base original de OpenML dentro de este proyecto."""
from pathlib import Path
import json
import pandas as pd
from sklearn.datasets import fetch_openml


def descargar_datos():
    raiz = Path(__file__).resolve().parents[1]
    ruta = raiz / 'data' / 'raw' / 'porto_seguro.csv'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    datos = fetch_openml(data_id=42742, as_frame=True,
                         data_home=str(raiz / '.cache' / 'openml'))
    X = datos.data
    y = datos.target
    df = X.copy()
    df['target'] = y
    df.to_csv(ruta, index=False)
    (raiz / 'data' / 'descripcion_openml.txt').write_text(datos.DESCR, encoding='utf-8')
    print('Archivo guardado:', ruta)
    print('Dimensiones descargadas:', df.shape)
    return ruta


if __name__ == '__main__':
    descargar_datos()
