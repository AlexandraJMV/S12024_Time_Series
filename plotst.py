import numpy      as np
import utility    as ut
import matplotlib.pyplot as plt
from train import acf_lags

# Load data from data.csv
def load_data_csv(horizon):
    path = f"est_{horizon}.csv"
    data = ut.load_data_csv(path)
    return data

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

def plot_regr(values, estimate, horizon):

    fig, ax = plt.subplots(figsize = (14,10))

    x_mean = np.mean(values)
    y_mean = np.mean(estimate)

    # Compute slope (m) and intercept (b) using least squares method
    slope = np.sum((values - x_mean) * (estimate - y_mean)) / np.sum((values - x_mean) ** 2)
    intercept = y_mean - slope * x_mean

    regression_line = slope * values + intercept

    # Graficar los datos y la línea de regresión
    ax.scatter(values, estimate, color='blue', label='Valores Estimados vs. Reales')
    ax.plot(values, regression_line, color = 'red', label='Regression Line')
    ax.set_xlabel('Target Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Model-Outputs vs. Targets')
    ax.legend()
    ax.grid(True)

    
    plt.savefig(f'fig3_{horizon}.png')
     
def plot_residual_acf(values, estimate, horizon):
    acf = np.array([acf_lags( (values - estimate) , i) for i in range(0, len(values))])

    fig, ax = plt.subplots(figsize = (14,10))
    ax.stem(acf)
    
    ax.axhline(y=0.2, color='r', linestyle='--')
    ax.axhline(y=-0.2, color='r', linestyle='--')
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Residual ACF')
    ax.grid(True)

    plt.savefig(f'fig2_{horizon}.png')

# Genrate figures
def gen_figures(X, horizon):
    values = X[:, 0]
    estimate = X[:, 1]

    plot_series( values, estimate, horizon)

    plot_residual_acf(values, estimate, horizon)

    plot_regr( values, estimate, horizon)

    return 

# Save  figures
def save_figures(X,Param):
    return 0


# Save Data 
def save_figures(X,Y):
    return 0


# Beginning ...
def main():      
    prop, memo, horizon = ut.load_conf()
    Data      = load_data_csv(horizon)	

    gen_figures(Data, horizon)    

    #save_figures()
    


if __name__ == '__main__':   
	 main()

