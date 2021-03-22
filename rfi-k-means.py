import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

from scipy import signal, sparse
from scipy.sparse.linalg import spsolve
from sklearn.cluster import MiniBatchKMeans
from seek.mitigation.sum_threshold import get_rfi_mask


def ArPLS(y, lam=1e4, ratio=0.05, itermax=10):
    '''
    copy from https://irfpy.irf.se/projects/ica/_modules/irfpy/ica/baseline.html
    
    Baseline correction using asymmetrically
    reweighted penalized least squares smoothing
    Sung-June Baek, Aaron Park, Young-Jin Ahna and Jaebum Choo,
    Analyst, 2015, 140, 250 (2015)

    Inputs:
        y:
            input data (i.e. SED curve)
        lam:
            parameter that can be adjusted by user. The larger lambda is,
            the smoother the resulting background, z
        ratio:
            wheighting deviations: 0 < ratio < 1, smaller values allow less negative values
        itermax:
            number of iterations to perform
    Output:
        the fitted background vector
    '''

    N = len(y)
    #  D = sparse.csc_matrix(np.diff(np.eye(N), 2))
    D = sparse.eye(N, format='csc')
    D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    D = D[1:] - D[:-1]

    D = D.T
    w = np.ones(N)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt
        
    return z


def removeBaseline(visibility):
	norm1stfit, norm2ndfit, BxHPF = visibility.copy(), visibility.copy(), visibility.copy()
	filteredDataNorm = np.abs(visibility - norm1stfit - norm2ndfit)
	for data, bsline1st, bsline2nd in [(visibility, norm1stfit, norm2ndfit)]:
		for i in range(np.shape(data)[0]):
			bsline1st[i,:] = np.abs(ArPLS(data[i,:], lam=1e2, ratio=0.08))
			bsline2nd[i,:] = np.abs(ArPLS(data[i,:] - bsline1st[i,:], lam=1e2, ratio=0.08))
			# 
			b, a = signal.butter(5, 1/1.5, 'high', analog=False)
			BxHPF[i,:] = signal.filtfilt(b, a, data[i,:], axis=-1)
	for i in range(np.shape(data)[0]):
		# Get moving average of size 5
		w = 5	# Size
		filteredDataNorm[i,:] = np.convolve(np.abs(visibility[i,:] - norm1stfit[i,:] - np.abs(BxHPF[i,:])), np.ones(w), 'same') / w
	return filteredDataNorm


def getRemask(data, rfiMask):
	rfiVis = np.abs(rfiMask+1)
	rfiMask = np.abs(rfiMask)
	rfiMask[rfiMask<1e4] = 1
	rfiMask[rfiMask>1e4] = np.inf
	remask = data.copy()
	remask[rfiMask==1] = 1
	remask = (np.log(rfiVis) + remask) / 2
	remask[remask <= 20] = 1
	remask[remask > 1] = np.inf
	return remask


def flagRFI(visibility):
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
		threshold = cluster_data[0][np.where(y_kmeans==y_kmeans[-1])[0][0]-6]
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


srcDir = '/home/stcm/Documents/HIRAX/hide/hide/data/1year/2016/03/21/'
#srcDir = '/data/path/to/hide/simulation/samples/data/1year/2016/03/21/'

for count, filename in enumerate(sorted(os.listdir(srcDir)), start=1):
	if filename.endswith(".h5"):
		with h5py.File(str(srcDir+filename), 'r') as fil:
			dataPhase0, dataPhase1 = np.array(fil['P/Phase0']), np.array(fil['P/Phase1'])
			rfiPhase0, rfiPhase1 = np.array(fil["RFI/Phase0"]), np.array(fil["RFI/Phase1"])
			
			data, rfiMask = dataPhase1, rfiPhase0

			absDATA = np.abs(data)

			vis = removeBaseline(absDATA)

			loggedVis = np.log(vis)

			rfiRemask = getRemask(loggedVis, rfiMask)


			'''K-Means Entire Space Flag Implementation'''
			print("\nStarting K-Means Full Window Flagging")
			visFlagged = flagRFI(loggedVis)
			plt.imshow(visFlagged, aspect='auto')
			plt.show()
			print("K-Means Full Window Flagging Complete\n\n")

