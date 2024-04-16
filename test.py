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