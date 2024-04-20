# AR's Training :

import numpy      as np
import utility    as ut

# Yule-Walker Method
def yule_walker(pinv_toepliz, acfv_T):
    # Cálculo de los coeficientes    
    return np.dot(pinv_toepliz, acfv_T)

# ACF for m-lags
def acf_lags(X, m):
    varianza = ut.acov(X, 0)

    if varianza == 0:
        sys.exit("Varianza 0, imposible calcular la autocorrelación")
        
    return ut.acov(X, m) / varianza
    
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
def train(x,y,memory, horizon):  
    # Parámetros por método Yule-Wlker
    # X = matriz autoregresiva
    # Y = valores de la serie
    
    AIC_vec = []
    N = len(y)
    
    # Probamos para cada valor de memoria m
    for m in range(1, memory + 1):
    
    # Estimación de parámetros a en Xa =Y 
        toepliz = mtx_toeplitz(y, m)       
        acfv    = np.array([acf_lags(y, i) for i in range(1, m + 1)])
        
        pinv_toepliz = pinv_svd(toepliz)    # Pseudo-inversa de la matriz de toepliz
        acfv_T       = acfv[:, np.newaxis]  # Traspuesta del arreglo valores de la función de auto correlación
        
        coefs = np.dot(pinv_toepliz, acfv_T)
        coefs = np.squeeze(coefs)
        
        # Hacemos una predicción
        pred = np.dot(x, coefs)
        
        # Calculamos el SSE
        SSE = ut.mean_squared_error(y, pred)
        
        # AIC
        
        AIC = np.log(SSE) + ( (2 * (m + 2)) / (N - m - 3) )
        AIC_vec.append(AIC)
        
    # Extraemos el indice de que tiene el menor AIC
    # memory final ; index delmenor AIC + 1

    return coefs

# Load data to train
def load_data_csv(horizon) -> np.array:
    
    path = f"trn_{horizon}.csv"  
    datos = ut.load_data_csv(path)
    
    # Separar X de Y
    X = datos[:, 1:]                # Matriz
    Y = np.squeeze(datos[:, 0])     # Vector 

    return X, Y

# Save coefficients 
def save_coef_csv(x, horizon):
    path = f"coef_{horizon}.csv"
    ut.write_csv(path, x, row = False)
    return
  
# Beginning ...
def main():
    param       = ut.load_conf()    
    
    memo, prop, horizon = param
    xe,ye         = load_data_csv(horizon)   
    
    coef        = train(xe,ye, memo, horizon)      
           
    save_coef_csv(coef, horizon)
       
if __name__ == '__main__':   
	 main()