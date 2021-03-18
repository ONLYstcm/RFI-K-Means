# Numpy dependencies
import numpy as np
# Sklearn dependencies
from sklearn.cluster import MiniBatchKMeans


def flag_rfi(visibility):
	# Initialise variables
	visuals_norm = np.round((visibility - np.min(visibility)) / (np.max(visibility) - np.min(visibility)), decimals=2)
	flaggedVisibility = visibility.copy()
	maximum, minimum = np.max(visuals_norm), np.min(visuals_norm)
	diffCluster = maximum - minimum
	std = 1
	# RFI-K-Means flagging
	while (std/np.sqrt(diffCluster) >= diffCluster):
		# Get unique data points in visibility with weights
		cluster_data = np.unique(visuals_norm, return_counts=True)
		countData = np.round(1000*(cluster_data[1]-np.min(cluster_data[1]))/(np.max(cluster_data[1]-np.min(cluster_data[1]))))
		# Create kmeans instance with number of clusters
		kmeans = MiniBatchKMeans(n_clusters=2, n_init=1, max_no_improvement=15)
		# Run K-Means algorithm
		y_kmeans = kmeans.fit_predict(cluster_data[0].reshape(-1, 1), sample_weight=countData)
		# Get new threshold which is the maximum value in the bottom cluster
		threshold = cluster_data[0][np.where(y_kmeans==y_kmeans[-1])[0][0]]
		std = np.std(cluster_data[0][np.where(y_kmeans==y_kmeans[-1])[0]])
		# Get index positions of RFI
		indexes = np.where(visuals_norm >= threshold)
		# Flag RFI
		flaggedVisibility[indexes] = np.inf
		# Set RFI samples to lower amplitude
		visuals_norm[indexes] = np.min(visuals_norm)
		# Get difference between cluster senters
		diffCluster = np.abs(np.diff(kmeans.cluster_centers_[:,0]))[0]
	return flaggedVisibility
