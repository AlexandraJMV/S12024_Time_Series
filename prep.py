import numpy      as np
from   csv    import reader, writer 
import utility    as ut

# Qur son interpretadores y como los usa el compu

# Load data from data.csv
def load_data_csv():
  path = "data.csv"
  datos = []

  with open(path, mode = 'r') as archivo_data:
    lector = reader( archivo_data )
    
    for linea in lector:
      dato = float(linea[0])
      datos.append(dato)
    
  return datos


# Create Auto-regressive-Matrix 
def ar_matrix(X,Param):
  
  # DUDA
  m, p = Param                # m : largo memoria, p : proporción train-test
  
  AR_X = []                   # Matriz autoregresiva
  cant_elem = len(X) - m      # Largo de cada vector 
  
  for index in range(cant_elem):
    
    # Construímos la matriz fila por fila
    row = [ X[i] for i in range( m + index, index - 1, -1 )] 
    AR_X.append(row)
    
  # Ahora separamos en training
  cant_data = len(AR_X)
  index = int( (cant_data * p) / 100 )
  
  train = AR_X[:index]
  test = AR_X[index:]
    
  return train, test

def write_csv(path, data):
  with open(path, mode = 'w', newline='') as archivo:
    
    data_writer = writer(archivo)
    for point in data:
      data_writer.writerow(point)
      
# Save Data 
def save_data_csv(X,Y):
  train_path = "trn_h.csv"
  test_path  = "tst_h.csv"
  
  write_csv(train_path, X)
  write_csv(test_path, Y)
  return


# Beginning ...
def main():        
    Param     = ut.load_conf()	
    Data      = load_data_csv()	
    
    Param = 3, 80
    Data = [1,2,3,4,5,6,7,8,9,10,11,12]
    
    dtrn,dtst = ar_matrix(Data, Param)  
    
    save_data_csv(dtrn,dtst)


if __name__ == '__main__':   
	 main()

