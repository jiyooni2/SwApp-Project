import matplotlib.pyplot as plt
import pandas as pd
import json

plt.figure(figsize=(19, 7))

plt.subplot(1, 2, 1)
with open("./data/seoul-polygons.json", "r") as seoul_json:
    seoul = json.load(seoul_json)
    for polygon in seoul:
        pCopy = polygon[:]
        pCopy.append(pCopy[0])
        xs, ys = zip(*pCopy)
        plt.plot(xs, ys, 'grey')
apt_combined_df = pd.read_csv("combined_df.csv")
plt.scatter(apt_combined_df["x"], apt_combined_df["y"], c=apt_combined_df["price"])
plt.jet()
plt.colorbar()
plt.clim(0, 500000)


plt.subplot(1, 2, 2)
with open("./data/seoul-polygons.json", "r") as seoul_json:
    seoul = json.load(seoul_json)
    for polygon in seoul:
        pCopy = polygon[:]
        pCopy.append(pCopy[0])
        xs, ys = zip(*pCopy)
        plt.plot(xs, ys, 'grey')
predicted_df = pd.read_csv("predicted_df.csv")
plt.scatter(predicted_df["x"], predicted_df["y"], c=predicted_df["price"])
plt.jet()
plt.colorbar()
plt.clim(0, 500000)


plt.show()