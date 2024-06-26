# My Utility : auxiliars functions

import numpy  as np
import pandas as pd
  
#load parameters from conf.csv
def load_conf():
    # porcentaje test-train , lag maximo : memoria maxima , horizonte : h
    path = "conf.csv"
    conf = load_data_csv(path, type = float)
    conf = conf[0], int(conf[1]), int(conf[2])

    return conf

# Measure
def mean_abs_error(yv, yp ):
  """
  yv : Valores verdaderos
  yp : Valores inferidos

  Calcula el MAE dados los valores reales e inferidos
  """
  
  N = len(yv)                   # Largo de la serie

  error = np.abs(yv - yp)    # Calculamos valor absoluto de los errores
  mae = np.sum(error)/ N     # Promediado del valor absoluto de los errores 
  return mae

def root_mean_squared_error(yv, yp ) :
  """
  yv : Valores verdaderos
  yp : Valores inferidos

  Calcula el rmse dados los valores reales e inferidos
  """
  return np.sqrt( mean_squared_error(yv, yp) )

def r_2(yv, yp):
  """
  yv : Valores verdaderos
  yp : Valores inferidos

  Calcula el R2 dados los valores reales e inferidos
  """
   
  error = yv - yp
  error_variance = np.var(error)

  true_variance = np.var(yv)

  return 1 - (error_variance / true_variance)

def modified_NS_efficiency(yv, yp ):

  mean = np.mean(yv)
  abs_error = np.abs( yv - yp )  
  abs_algo = np.abs( yv - mean )

  factor = np.sum(abs_error) / np.sum(abs_algo)

  return 1 - factor

def metricas(x , y):
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

def squared_error(yv, yp) -> float:
  """
  yv : Valores verdaderos
  yp : Valores inferidos

  Calcula el mse dados los valores reales e inferidos
  """
  error = np.sum((yv - yp) ** 2)
  return error

def mean_squared_error(yv, yp ) :
  """
  yv : Valores verdaderos
  yp : Valores inferidos

  Calcula el mse dados los valores reales e inferidos
  """
  
  return squared_error(yv, yp) / len(yv)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Auto covarianza 
def acov(series_data, lag ):
    
    """
    series_data : numpy array con los datos de la serie de tiempo
    lag         : Valor entero que indica el lag entre los valores para el calculo de la auto-covarianza

    Calcula la auto-covarianza de una serie de tiempo para un cierto lag 
    """
    N   = len(series_data)
    mean = series_data.mean()

    sum = np.sum( (series_data[: N - lag] - mean) * (series_data[lag:] - mean) ) 
    return sum / (N - lag)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Lectura y escritura de datos

def write_csv(path , data , row  = True):
  """
  path : nombre del archivo
  data : lista de datos a escribir 
  row  : Indicador si data es una lista de listas

  Escribe un arreglo o matriz datos en un archivo CSV.
  Cada elemento de la lista debe ser una lista de lo que se desea que este en una linea
  """
  if not row : data = [data]
  
  df = pd.DataFrame(data)
  df.to_csv(path, index = False, header=False)
  
def load_data_csv(path , type = float):
  """
  path : Nombre del archivo a leer

  Lee un archivo csv guardando cada una de sus lineas en un arreglo. 
  Transforma cada elemento leido en un float 
  """
  data = np.loadtxt(path, delimiter= ",", dtype = type)
  return data

