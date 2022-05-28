import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(19, 10), subplot_kw={"projection": "3d"})

apt_combined_df = pd.read_csv("combined_df.csv")
X = apt_combined_df["x"].to_numpy()
Y = apt_combined_df["y"].to_numpy()
Z = apt_combined_df["price"].to_numpy()

surf = ax[0].plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
ax[0].set_zlim(0, 500000)

predicted_df = pd.read_csv("predicted_df.csv")
X = predicted_df["x"].to_numpy()
Y = predicted_df["y"].to_numpy()
Z = predicted_df["price"].to_numpy()

surf = ax[1].plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
ax[1].set_zlim(0, 500000)

plt.show()
