import numpy as np
from matplotlib import pyplot as plt

file = 'data/hyper_elastic/test_err.npy'
err = np.load(file)
# print(err[:10])
# statistics
max_err = np.max(err)
min_err = np.min(err)
mean_err = np.mean(err)
err[err<0]=0
print('max_err={:.4f}%, min_err={:.4f}%, mean_err={:.4f}%'.format(max_err, min_err, mean_err))
plt.hist(err)
plt.grid(True)
plt.xlabel('relative error (%)')
plt.show()
# histogram
# kde = stats.gaussian_kde(err)
# x = np.linspace(min_err*0.3, max_err*1.05, 100)
# p = kde(x)
# plt.figure(figsize=[7,5])
# plt.plot(x, p)
# plt.xlabel('err (%)')
# plt.ylabel('pdf')
# plt.grid(True)
# plt.show()
