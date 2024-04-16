import numpy as np
import train as tr


matrix = [
    [0.9794, 1.8407],
    [0.6790, 0.1054],
    [1.9033, 1.4757]
]

time_series = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

U, S, Vh = np.linalg.svd(matrix, full_matrices = False)

print(tr.pinv_svd(matrix))
print(tr.acf_lags(time_series, 0))

print()

print(tr.mtx_toeplitz(time_series, m = 4))

import numpy as np
from scipy.linalg import toeplitz

# Example autocovariance values
autocovariance = [1.0, 0.8, 0.6, 0.4, 0.2]

# Create a Toeplitz matrix using autocovariance values
toeplitz_matrix = toeplitz(autocovariance)

print("Toeplitz Matrix:")
print(toeplitz_matrix)