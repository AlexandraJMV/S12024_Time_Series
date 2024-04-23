# AR's Training :
import numpy      as np
import utility    as ut
from prep import ar_matrix
from test import fordward
import matplotlib.pyplot as plt

def plot_series(values , estimate  , horizon) :

    fig, ax = plt.subplots(figsize = (14,10))
    ax.plot(values, color = 'blue', label = 'Real-Value')
    ax.plot(estimate, color = 'red', label = 'Estimated value')

    fig.suptitle(f'Testing : Horizon = {horizon}-step')

    ax.set_xlabel('samples')
    ax.set_ylabel('values')
    ax.legend()

    plt.savefig(f'fig1_{horizon}.png')
    return 


# Yule-Walker Method
def yule_walker(pinv_toepliz, acfv_T):
    # Cálculo de los coeficientes    
    return np.dot(pinv_toepliz, acfv_T)

# ACF for m-lags
def acf_lags(X, m):
    varianza = ut.acov(X, 0)
    return ut.acov(X, m) / varianza
    
# Toeplitz matrix
def mtx_toeplitz(X, m):
    
    toeplitz = np.array([])
    flag = False 
    count = 0

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
    
        count +=1

    # Si es que m = 1, se asegura que devuelva una matriz
    if count == 1 :
        toeplitz = np.expand_dims(toeplitz, axis=0)

    return toeplitz

# Pseudo-inverse by use SVD
def pinv_svd(matrix):
    # Descomponemos la matriz
    U, S, Vh = np.linalg.svd(matrix, full_matrices = False)
    
    U_T      = U.T               # Traspuesta de U
    S_inv    = np.linalg.inv(np.diag(S))  # Inversa de la matriz S
    
    # Armamos la matriz inversa
    pinv_matrix = np.dot( Vh , np.dot( S_inv, U_T ) )

    return pinv_matrix

def construct_yule_walker(series, memory):
    toepliz = mtx_toeplitz(series, memory) 
    acfv    = np.array([acf_lags(series, i) for i in range(1, memory + 1)])
    
    pinv_toepliz = pinv_svd(toepliz)    # Pseudo-inversa de la matriz de toepliz
    return pinv_toepliz, acfv

#AR's Training 
def train(y,memory, horizon):  
    # Parámetros por método Yule-Wlker

    # Y = valores de la serie
    # Memory = Memoria maxima
    # Horizon = horizonte de prdiccion
    
    AIC_vec = []
    coefs_vec = []
    error = []

    # Probamos para cada valor de memoria m
    for m in range(1, memory + 1):

        # Creacion de la matriz regresora
        series, matrix = ar_matrix(y, m, horizon)
        N = len(series)

        # Creamos la matriz de toeplitz y el vector relevantes
        toe, acfv = construct_yule_walker(series, m)

        # Calculamos los coeficientes
        coefs = yule_walker(toe, acfv)
        coefs_vec.append(coefs)
        
        # Hacemos una predicción
        pred = fordward(matrix, coefs)
        pred = np.squeeze(pred)

        plot_series(series, pred, m)

        # Calculamos el SSE
        SSE = ut.squared_error(series, pred)
        error.append( ut.mean_squared_error(series, pred) )

        # AIC
        AIC = np.log(SSE) + ( (2 * (m + 2) / ( N - m - 3)) )
        BIC = 2 * np.log( SSE / (N - m)) + ((m / N)* np.log(N))
        AIC_vec.append(BIC)
        
    # Extraemos el indice de que tiene el menor AIC
    # memory final ; index del menor AIC + 1

    best = np.argmin(AIC_vec, axis = 0)
    best_coefs = np.squeeze(coefs_vec[best])

    return best_coefs

# Load data to train
def load_data_csv(horizon) :
    
    path = f"trn_{horizon}.csv"  
    datos = ut.load_data_csv(path)
    
    # Separar X de Y
    Y = np.squeeze(datos)     # Vector 

    return Y

# Save coefficients 
def save_coef_csv(x, horizon):
    path = f"coef_{horizon}.csv"
    ut.write_csv(path, x, row = False)
    return
  
# Beginning ...
def main():
    param       = ut.load_conf()    
    
    prop, memo, horizon = param

    # ye = serie 
    ye         = load_data_csv(horizon)  
    coef        = train(ye, memo, horizon)

    save_coef_csv(coef, horizon)
       
if __name__ == '__main__':   
	 main()