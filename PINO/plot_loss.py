import numpy as np
from matplotlib import pyplot as plt 
loss = np.load('loss_info/hyper_elastic/training.npy')
plt.figure(figsize=(5,4))
plt.semilogy(loss)
plt.grid(True)
plt.xlabel('epoch #')
plt.ylabel('loss')
plt.show()