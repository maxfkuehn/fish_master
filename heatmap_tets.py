import matplotlib.pyplot as plt
import numpy as np

heat_map_matrix_single = np.array([[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9]])

plt.pcolormesh(heat_map_matrix_single, cmap='hot')
plt.colorbar()
plt.show()