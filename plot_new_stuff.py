import csv
import numpy as np

# from scipy.fft import fft, fftfreq
import math
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import itertools

from matplotlib.colors import LinearSegmentedColormap, ListedColormap

def multi_sqrt(grid):
    for _ in range(2):
        grid = np.sqrt(grid)
    return grid

def rmse(grid):
    return np.sqrt(np.nanmean((grid)**2, axis=2))


grid_euc_distance = np.load("datasets/grid_euc_distance.npy")
grid_abs_distance = np.load("datasets/grid_abs_distance.npy")
grid_mse = np.load("datasets/grid_mse.npy")

grid = grid_euc_distance
print(grid)

plt.figure(figsize=(11,10), layout="constrained")

grid[grid == 0] = np.nan

grid = grid/(15+grid)
grid = np.power(100,grid)
# grid = (np.log10(grid))
# grid[grid == -np.inf] = np.nan
# plt.imshow((np.nanstd((grid), axis=2)))

# plt.imshow(np.log10(np.std(grid, axis=2)))

# grid[grid > 10] = 10

# grid_rem_0 = grid.copy()
# grid = np.log10((grid))
# grid_rem_1 = grid.copy()
# second_smallest = np.nanmin(grid[grid != np.nanmin(grid)])
# if np.nanmin(grid) < 0:
#     grid -= np.nanmin(grid)-((second_smallest-np.nanmin(grid))/2)
# elif np.nanmin(grid) > 0:
#     grid += np.nanmin(grid)+((second_smallest-np.nanmin(grid))/2)
# else:
#     grid += ((second_smallest-np.nanmin(grid))/2)

# grid = np.log10((grid))
# grid_rem_2 = grid.copy()
# second_smallest = np.nanmin(grid[grid != np.nanmin(grid)])
# if np.nanmin(grid) < 0:
#     grid -= np.nanmin(grid)-((second_smallest-np.nanmin(grid))/2)
# elif np.nanmin(grid) > 0:
#     grid += np.nanmin(grid)+((second_smallest-np.nanmin(grid))/2)
# else:
#     grid += ((second_smallest-np.nanmin(grid))/2)

# grid = np.log10(grid)
# grid_rem_3 = grid.copy()

# grid -= np.nanmin(grid)

# # grid = 1/(grid)
# grid_rem_4 = grid.copy()
# grid[grid > np.nanmean(grid)] = np.nan
# grid = multi_sqrt(grid)

# grid = np.log10(grid)






base_colors = [
    "#FF5733", "#5733FF",  # Orange shades
    "#5733FF", "#FFC300",  # Purple shades
    "#FFC300", "#33FFF5",  # Brown shades
    "#33FFF5", "#FF33A8",  # Pink shades
    "#FF33A8", "#8D33FF",  # Gray shades
    "#8D33FF", "#33FF85",  # Yellow shades
    "#33FF85", "#FF8C33",   # Teal shades
    "#FF8C33", "#33C3FF"   # Teal shades
]   

# Step 2: Create gradients for each base color set
n_shades = 10  # Number of gradients per color
gradients = []

for i in range(0, len(base_colors), 2):
    start_color = base_colors[i]
    end_color = base_colors[i + 1]
    cmap = LinearSegmentedColormap.from_list(f"gradient{i//2}", [start_color, end_color], N=n_shades)
    gradients.extend(cmap(np.linspace(0, 1, n_shades)))

# Step 3: Create a ListedColormap from all gradients
full_colormap = ListedColormap(gradients)


grid = (grid -np.nanmin(grid)) / (np.nanmax(grid)-np.nanmin(grid))

# plt.imshow(grid, extent=[-2,2,2,-2], cmap=full_colormap)
plt.imshow(grid, extent=[-2,2,2,-2], cmap="viridis")
plt.xlabel("V1")
plt.ylabel("V2")
plt.colorbar()
plt.gca().invert_yaxis()
plt.savefig("grid.png")
plt.clf()



# Histograms
figs, axs = plt.subplots(6,1,figsize=(10,15), layout="constrained")

grid_rem_0 = (grid_rem_0 -np.nanmin(grid_rem_0)) / (np.nanmax(grid_rem_0)-np.nanmin(grid_rem_0))
grid_rem_1 = (grid_rem_1 -np.nanmin(grid_rem_1)) / (np.nanmax(grid_rem_1)-np.nanmin(grid_rem_1))
grid_rem_2 = (grid_rem_2 -np.nanmin(grid_rem_2)) / (np.nanmax(grid_rem_2)-np.nanmin(grid_rem_2))
grid_rem_3 = (grid_rem_3 -np.nanmin(grid_rem_3)) / (np.nanmax(grid_rem_3)-np.nanmin(grid_rem_3))
grid_rem_4 = (grid_rem_4 -np.nanmin(grid_rem_4)) / (np.nanmax(grid_rem_4)-np.nanmin(grid_rem_4))
axs[0].hist(grid_rem_0.flatten(), bins=200, alpha=0.5, label="Incoming")
axs[0].set_title("Incoming")
axs[1].hist(grid_rem_1.flatten(), bins=200, alpha=0.5, label="Change 1")
axs[1].set_title("Change 1")
axs[2].hist(grid_rem_2.flatten(), bins=200, alpha=0.5, label="Change 2")
axs[2].set_title("Change 2")
axs[3].hist(grid_rem_3.flatten(), bins=200, alpha=0.5, label="Change 3")
axs[3].set_title("Change 3")
axs[4].hist(grid_rem_4.flatten(), bins=200, alpha=0.5, label="Change 4")
axs[4].set_title("Change 4")

axs[5].hist(grid.flatten(), bins=200, alpha=0.5, label="Final")
axs[5].set_title("Final")

plt.savefig("hist.png")