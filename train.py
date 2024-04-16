# AR's Training :

import numpy      as np
import utility    as ut
import sys
from csv import reader
from prep import write_csv

# Yule-Walker Method
def yule_walker(pinv_toepliz, acfv_T):
    # Cálculo de los coeficientes    
    return np.dot(pinv_toepliz, acfv_T)

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
        
    if k == N:
        sys.exit("Division por 0, lag = tamaño de la serie")
    
    return sum / (N - k)

# ACF for m-lags
def acf_lags(X, m):
    varianza = acov(X, 0)

    if varianza == 0:
        sys.exit("Varianza 0, imposible calcular la autocorrelación")
        
    return acov(X, m) / varianza
    
# Toeplitz matrix
def mtx_toeplitz(X, m):
    
    toeplitz = np.array([])
    flag = False 

    for k in range(m, 0, -1):          # m filas
        
        row = []
        for i in range(m - k, 0 , -1):          # Rellena hasta el diagonal (sin contar el diagonal)
            row.append( acf_lags(X, i) )
        
        for j in range(0, k):               # Rellena el resto de forma secuencial
            row.append( acf_lags(X, j) )
        
        row = np.array(row)

        if flag : toeplitz= np.vstack( (toeplitz, row) )

        if not flag:
            toeplitz= np.hstack( (toeplitz, row) )
            flag = True


    
    return toeplitz

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
    # Parámetros por método Yule-Wlker
    # X = matriz autoregresiva
    # Y = valores de la serie
    m, p = param                   # memoria, prop
    
    toepliz = mtx_toeplitz(y, m)
    acfv    = np.array([acf_lags(y, i) for i in range(1, m + 1)])
    
    pinv_toepliz = pinv_svd(toepliz)    # Pseudo-inversa de la matriz de toepliz
    acfv_T       = acfv[:, np.newaxis]  # Traspuesta del arreglo valores de la función de auto correlación

    print(toepliz)
    print(acfv.shape)
    print(acfv[:, np.newaxis])

    coefs = np.dot(pinv_toepliz, acfv_T)
    coefs = np.squeeze(coefs)

    return coefs

# Load data to train
def load_data_csv():
    path = "trn_h.csv"
    datos = np.array([])
    flag = False

    with open(path, mode = 'r') as archivo_data:
        lector = reader( archivo_data )
    
        for linea in lector:                
            dato = np.array([float(i) for i in linea])
            
            if flag : datos = np.vstack( (datos, dato) )

            if not flag:
                datos = np.hstack( (datos, dato) )
                flag = True

    
    # Separar X de Y
    X = datos[:, 1:]                # Matriz
    Y = np.squeeze(datos[:, 0])     # Vector 

    return X, Y

# Save coefficients 
def save_coef_csv(x):
    write_csv("coef_h.csv", [x])
    return
  
# Beginning ...
def main():
    param       = ut.load_conf()    # Supongo que el largo de esa memoria        
    xe,ye       = load_data_csv()   
    coef        = train(xe,ye,param)             
    save_coef_csv(coef)
       
if __name__ == '__main__':   
	 main()
