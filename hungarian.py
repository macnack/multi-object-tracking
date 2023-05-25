import numpy as np
from scipy.optimize import linear_sum_assignment

adj_matrix = np.array([[108, 125, 150], [150, 135, 175], [122, 148, 250]])
row_idx, col_idx = linear_sum_assignment(adj_matrix)

print(col_idx)
print(row_idx)
print(adj_matrix[col_idx, row_idx])