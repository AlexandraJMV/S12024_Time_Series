import numpy as np
import utility as ut
from csv import reader
from train import load_data_csv as load

def load_data_csv():
	X, Y = load("tst_h.csv")
	return X,Y

def load_coef_csv():
	path = "coef_h.csv"
	coefs = []
      
	with open(path, mode = 'r') as archivo_coef:
		lector = reader( archivo_coef )
            
		for linea in lector:
			for coef in linea:
				coefs.append(float(coef))
	return np.array(coefs)

def fordward(x: np.array ,a: np.array):
    return np.dot(x, a)

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

	List   = ut.metricas(yv,zv) 		# Faltan las metricas
	save_measure_csv([1], yv, zv)

if __name__ == '__main__':   
	 main()