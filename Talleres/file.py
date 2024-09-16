import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data with make_blobs
X, _ = make_blobs(n_samples=300, centers=5, cluster_std=1.0, random_state=42)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Get the cluster centers
centers = kmeans.cluster_centers_

# Create scatter plot of the clustered data
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Plot cluster centers with larger points
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')

# Add some style
plt.gca().set_facecolor('black')
plt.title('Artistic KMeans Clustering', color='white')
plt.xticks([])
plt.yticks([])
plt.show()
