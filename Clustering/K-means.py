from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

# create model
kmeans = KMeans(n_clusters=2, random_state=0)

# fit
kmeans.fit(X)

print("cluster centre:", kmeans.cluster_centers_)
print("labels of every point:", kmeans.labels_)

# visualisation
for i in range(2):
    cluster_points = X[kmeans.labels_ == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i+1}")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="red", marker="x", label="Centroids")
plt.legend()
plt.show()    