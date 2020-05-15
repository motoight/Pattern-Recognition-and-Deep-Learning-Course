import numpy as np
import torch
import matplotlib.pyplot as plt
np.random.seed(2)

T = 20
L = 1000
N = 100

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float64')
# for i in range(5):
#     plt.plot(data[i])
# plt.show()
# torch.save(data, open('traindata.pt', 'wb'))