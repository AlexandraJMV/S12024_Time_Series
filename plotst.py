import numpy      as np
import utility    as ut
import matplotlib.pyplot as plt

# Load data from data.csv
def load_data_csv():
    path = "est_h.csv"
    data = ut.load_data_csv(path)
    return data

def plot_series(values : np.array, estimate : np.array ) -> None :
    title = "Valor real vs. valor estimado"

    fig, ax = plt.subplots()
    ax.plot(values, color = 'blue', label = 'Real-Value')
    ax.plot(estimate, color = 'red', label = 'Estimated value')

    fig.suptitle('Testing : Horizon : ????')

    ax.set_xlabel('samples')
    ax.set_ylabel('values')
    ax.legend()

    plt.savefig('fig1_h.png')
    return 

# Genrate figures
def gen_figures(X,Param):

    

    return 0

# Save  figures
def save_figures(X,Param):
    return 0


# Save Data 
def save_figures(X,Y):
    return 0


# Beginning ...
def main():            
    Data      = load_data_csv()	
    #gen_figures()    
    #save_figures()
    


if __name__ == '__main__':   
	 main()

