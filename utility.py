# My Utility : auxiliars functions

import numpy  as np
from csv import writer, reader 
import sys
  
#load parameters from conf.csv
def load_conf():
    # memoria, proporcion test-train 
    return 3, 80

# Measure
def mean_abs_error(yv : np.array, yp : np.array)->float:
  """
  yv : Valores verdaderos
  yp : Valores inferidos

  Calcula el MAE dados los valores reales e inferidos
  """
  N = len(yv)                   # Largo de la serie

  error = np.abs(yv - yp)    # Calculamos valor absoluto de los errores
  mae = np.sum(error)/ N     # Promediado del valor absoluto de los errores 
  return mae

def root_mean_squared_error(yv : np.array, yp :np.array) -> float :
  """
  yv : Valores verdaderos
  yp : Valores inferidos

  Calcula el rmse dados los valores reales e inferidos
  """
  N = len(yv)

  error = np.square(yv - yp)
  mse = np.sum(error) / N

  return np.sqrt(mse)

def r_2(yv : np.array , yp : np.array) -> float:
  """
  yv : Valores verdaderos
  yp : Valores inferidos

  Calcula el R2 dados los valores reales e inferidos
  """
   
  error = yv - yp
  error_variance = np.var(error)

  true_variance = np.var(yv)

  return 1 - (error_variance / true_variance)

def modified_NS_efficiency(yv : np.array, yp : np.array) -> float :

  mean = np.mean(yv)
  abs_error = np.abs( yv - yp )  
  abs_algo = np.abs( yv - mean )

  factor = np.sum(abs_error) / np.sum(abs_algo)

  return 1 - factor

def metricas(x : np.array, y: np.array)->list[float]:
    """
    x : Valores reales de una serie de tiempo
    y : Valores estimados de una serie de tiempo

    Calcula las metricas MAE, RMSE, R2 Y mNSE

    Retorna : 
      Lista con cada una de estas metricas (en el orden escrito)
    """
    return [mean_abs_error(x, y), 
            root_mean_squared_error(x, y),
            r_2(x, y),
            modified_NS_efficiency(x, y)]

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Auto covarianza 
def acov(series_data : np.array, lag : int)-> float:
    
    """
    series_data : numpy array con los datos de la serie de tiempo
    lag         : Valor entero que indica el lag entre los valores para el calculo de la auto-covarianza

    Calcula la auto-covarianza de una serie de tiempo para un cierto lag 
    """

    sum = 0
    N   = len(series_data)
    mean = series_data.mean()
    
    for i in range(N - lag):
        sum += (series_data[i] - mean) * (series_data[i + lag] - mean)
        
    if lag == N:
        sys.exit("Division por 0, lag = tamaño de la serie")
    
    return sum / (N - lag)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Lectura y escritura de datos

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

def load_data_csv(path : str) -> None :
  """
  path : Nombre del archivo a leer

  Lee un archivo csv guardando cada una de sus lineas en un arreglo. 
  Apila cada linea verticalmente
  
  Transforma cada elemento leido en un float 
  """
   
  datos = np.array([])       # Almacenamiento de datos 
  flag = False

  with open(path, mode = 'r') as archivo_data:
    lector = reader( archivo_data )

    for linea in lector:
      dato = np.array([float(i) for i in linea])

      if flag : datos = np.vstack( (datos, dato) )

      if not flag:
        datos = np.hstack( (datos, dato) )
        flag = True

  return datos

