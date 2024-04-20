import numpy      as np
import utility    as ut

# Que son interpretadores y como los usa el compu
# Terminado ahora si xd

# Load data from data.csv
def load_data_csv():
  path = "data.csv"
  datos = ut.load_data_csv(path)
    
  return datos


# Create Auto-regressive-Matrix 
def ar_matrix(X : np.ndarray, proportion : int, memory : int , horizon : int):

  # Matriz autoregresiva
  AR_X = np.array([])
  
  # Cantidad de elementos del set de datos final                                       
  cant_elem = len(X) - memory  - (horizon - 1)
  flag = False
  
  for index in range(cant_elem):
    
    # Extraemos el elemento que corresponde al valor a estimar
    yt = np.array([ X[ memory + horizon + index - 1 ] ])
    
    # Construimos el resto de la matriz
    vector_regresor = np.array([ X[i] for i in range( memory + index - 1, index - 1, -1 )])
    
    # Construimos la fila
    row = np.hstack(( yt, vector_regresor ))

    # Construimos la matriz
    if not flag :
      AR_X = np.hstack(( AR_X, row ))
      flag = True
    else:
      AR_X = np.vstack(( AR_X, row ))
      
  # Ahora separamos en training y testing
  cant_data = len(AR_X)
  index = int( (cant_data * proportion) / 100 )
  
  train = AR_X[:index]
  test = AR_X[index:]
  
  return train, test

# Save Data 
def save_data_csv(X : np.ndarray , Y : np.ndarray, h : int) -> None :
  """
  Guarda los datos de entrenamiento y testeo para un horizonte H
  
  Args:
    X (np.ndarray) : Datos de entrenamiento
    Y (np.ndarray) : Datos de testeo
    h (int)        : Horizonte de predicción
  """
  
  train_path = f"trn_{h}.csv"
  test_path  = f"tst_{h}.csv"
  
  ut.write_csv(train_path, X)
  ut.write_csv(test_path, Y)
  
  return

# Beginning ...
def main():        
    Param     = ut.load_conf()	
    Data      = load_data_csv()	
    
    memo, prop, horizon = Param
    Data = [1,2,3,4,5,6,7,8,9,10,11,12]
    
    dtrn,dtst = ar_matrix(Data, prop, memo, horizon) 
    save_data_csv(dtrn, dtst, horizon)


if __name__ == '__main__':   
	 main()

