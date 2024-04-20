import numpy as np
import utility as ut

def load_data_csv():
	path= "tst_h.csv"
	datos = ut.load_data_csv(path)

	# Separar X de Y
	X = datos[:, 1:]                # Matriz
	Y = np.squeeze(datos[:, 0])     # Vector 

	return X,Y

def load_coef_csv():
	path = "coef_h.csv"
	data = ut.load_data_csv(path)
	
	return data

def fordward(x: np.array ,a: np.array):
    return 

def save_measure_csv(metricas : list, y_true: np.array, y_pred: np.array):

	# Escritura de metricas
	ut.write_csv("metrica_h.csv", metricas, row = False)

	# Escritura valor real vs valor predicho
	data = np.column_stack( (y_true, y_pred) )
	ut.write_csv("est_h.csv", data)

	return

# Beginning ...
def main():			
	xv,yv  = load_data_csv()
	a      = load_coef_csv()
	zv     = fordward(xv,a)   

	List   = ut.metricas(yv,zv) 	
	save_measure_csv(List, yv, zv)

if __name__ == '__main__':   
	 main()