# AR's Training :

import numpy      as np
import utility    as ut
import sys
from csv import reader

# Yule-Walker Method
def yule_walker():
    
    # Parámetros por método Yule-Wlker
    toepliz = mtx_toeplitz()
    acfv    = acf_lags()
    
    pinv_toepliz = pinv_svd(toepliz)    # Pseudo-inversa de la matriz de toepliz
    acfv_T       = acfv_T.T             # Traspuesta del arreglo valores de la función de auto correlación
    
    # Cálculo de los coeficientes
    coef = np.dot(pinv_toepliz, acfv_T)
    
    return coef 

# Auto covarianza
def acov(X, k):
    """
    X : data
    k : lags
    """
    sum = 0
    N   = len(X)
    mean = X.mean()
    
    for i in range(N - k):
        sum += (X[i] - mean) * (X[i + k] - mean)
    
    return sum / (N - k)

# ACF for m-lags
def acf_lags(X, m):
    varianza = acov(X, 0)

    if varianza == 0:
        sys.exit("Varianza 0, imposible calcular la autocorrelación")
        
    return acov(X, m) / varianza
    
# Toeplitz matrix
def mtx_toeplitz():
    return 0

# Pseudo-inverse by use SVD
def pinv_svd(matrix):
    # Descomponemos la matriz
    U, S, Vh = np.linalg.svd(matrix, full_matrices = False)
    
    U_T      = U.T               # Traspuesta de U
    S_inv    = np.diag(1.0 / S)  # Inversa de la matriz S
    
    # Armamos la matriz inversa
    pinv_matrix = np.dot( Vh , np.dot( S_inv, U_T ) )
    
    return pinv_matrix

#AR's Training 
def train(x,y,param):        
    return 0


# Load data to train
def load_data_csv():
    path = "trn_h.csv"
    datos = []

    with open(path, mode = 'r') as archivo_data:
        lector = reader( archivo_data )
    
        for linea in lector:
            dato = [float(i) for i in linea]
            datos.append(dato)
    datos = np.matrix(datos)
    
    # Separar X de Y
    X = datos[:, 1:]
    Y = datos[:, 0]
    
    return X, Y

# Save coefficients 
def save_coef_csv(x):
  return 0
  
# Beginning ...
def main():
    #param       = ut.load_conf()            
    xe,ye       = load_data_csv()   
    #coef        = train(xe,ye,param)             
    #save_coef_csv(coef)
       
if __name__ == '__main__':   
	 main()
