import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

x = 2 * np.pi * np.linspace(-1, 1, 100)

z = np.sin(x.reshape(1, -1) + x.reshape(-1, 1))

cmap = np.zeros([256, 4])
cmap[:, 3] = np.linspace(0, 1, 256)
cmap = ListedColormap(cmap)

plt.figure()
# plt.pcolormesh(z + 1, cmap='bwr', edgecolors=None)
plt.pcolormesh(np.fliplr(z), cmap=cmap, edgecolors=None)
plt.savefig("temp")