import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib as mpl
# Create a custom colormap from black to white
colors = [(0, 0, 0), (1, 1, 1)]  # Black to white
# cmap = mcolors.LinearSegmentedColormap.from_list("CustomMap", colors)

cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                           ['black', 'white'],
                                           256)
data = np.random.rand(10, 10)
print(data)
# Generate some data (replace this with your MAP['map'])


# Define the boundaries and norm
bounds = [0, 1]  # Define appropriate boundaries, e.g., 0 and 1
# norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Plot the map
fig, ax = plt.subplots()
img = ax.imshow(data, interpolation='nearest',
                    cmap = cmap,
                    origin='lower')

# Add colorbar
cbar = plt.colorbar(img, cmap=cmap, boundaries=bounds, ticks=[0, 1])

plt.title("Gradient from black to white")
plt.show()
