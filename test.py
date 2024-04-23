import numpy as np
import utility as ut
from prep import ar_matrix

def load_data_csv(horizon ):

	path= f"tst_{horizon}.csv"
	datos = ut.load_data_csv(path)

	# Separar X de Y
	X = np.squeeze(datos)              # Vector, la serie

	return X

def load_coef_csv(horizon ):
	path = f"coef_{horizon}.csv"
	data = ut.load_data_csv(path)
	
	return data

def fordward(x,a):
	# dependiendo del tamano del arreglo de coeficientes, debemos adaptar la memoria de la matriz autoregresiva
	return np.dot(x, a)

def save_measure_csv(metricas , y_true, y_pred, horizon ):

	path_metricas = f"metrica_{horizon}.csv"
	path_est = f"est_{horizon}.csv"

	# Escritura de metricas
	ut.write_csv(path_metricas, metricas, row = False)

	# Escritura valor real vs valor predicho
	data = np.column_stack( (y_true, y_pred) )
	ut.write_csv(path_est, data)

	return

# Beginning ...
def main():		
	prop, memo, horizon = ut.load_conf()

	# Cargamos los coeficientes
	a     = load_coef_csv(horizon)

	# Calculamos la memoria
	memory_final = len(a)

	# Cargamos la serie de testing
	yv  = load_data_csv(horizon)

	# Creamos la matriz autoregresora
	yv, xv = ar_matrix(yv, memory_final, horizon)

	# Calculamos la prediccion
	zv     = fordward(xv,a)

	# Calculamos las metricas
	List   = ut.metricas(yv,zv)

	# Guardamos
	save_measure_csv(List, yv, zv, horizon)

if __name__ == '__main__':   
	 main()