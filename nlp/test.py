__author__ = 'jeff'

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(19680801)
# x = np.random.randn(3000000)
k = np.random.rand(1000, 30000)
# x = np.random.rand(300000)

x = k.reshape(30000000)
m = k < 0
n = k > 1
print(m.sum())
print(n.sum())
print(x.sum())

# example data
# mu = 100  # mean of distribution
# sigma = 15  # standard deviation of distribution
# x = mu + sigma * np.random.randn(437)

bins_list = np.array([0 + i * 0.1 for i in range(10)])

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, bins=bins_list)

# add a 'best fit' line
# y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
#      np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
# ax.plot(bins, y, '--')
# ax.set_xlabel('Smarts')
# ax.set_ylabel('Probability density')
# ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()

