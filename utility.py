# My Utility : auxiliars functions

import numpy  as np
from csv import writer
  
#load parameters from conf.csv
def load_conf():
    # memoria, proporcion test-train 
    return 3, 80

# Measure
def metricas(x,y):
    return 0

def write_csv(path : str, data : list[any], row : bool = True) -> None:
  """
  path : nombre del archivo
  data : lista de datos a escribir 
  row  : Indicador si data es una lista de listas

  Escribe una linea de datos en un archivo CSV.
  Cada elemento de la lista 'data' debe ser una lista con datos por linea
  """
  if not row : data = [data]

  with open(path, mode = 'w', newline='') as archivo:

    data_writer = writer(archivo)
    for point in data:
      data_writer.writerow(point)