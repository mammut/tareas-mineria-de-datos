#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from time import time

import numpy
import sklearn
import scipy
from sklearn import metrics


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets

from sklearn.cluster import KMeans, Ward, AgglomerativeClustering, DBSCAN, spectral_clustering
from sklearn import manifold
import skfuzzy

global CLUSTERS
CLUSTERS = 3

def conv(list1, list2):
	final_list = []
	for elem1 in list1:
		for elem2 in list2:
			if type(elem1) is list:
				final_list += [elem1+[elem2]]
			else:
				final_list += [[elem1, elem2]]
	return final_list


class Cluster:
	TITLE = None
	D3 = False

	def __init__(self, X, Y):
		self.X = X
		self.Y = Y

		self.params = {"n_clusters" : CLUSTERS}
		self.testing_params = {"n_clusters" : [CLUSTERS]}

	def run(self):
		return self.Y

	def report(self):
		start_time = time()
		Y_ = self.run()
		elapsed_time = time() - start_time

		values = {i : {e : 0 for e in set(Y_)} for i in set(self.Y)}
		for i in range(len(Y)):
			values[self.Y[i]][Y_[i]] += 1
		error1 = len(Y) - sum([max(value.values()) for value in values.values()])


		values = {i : {e : 0 for e in set(self.Y)} for i in set(Y_)}
		for i in range(len(Y)):
			values[Y_[i]][self.Y[i]] += 1
		error2 = len(Y) - sum([max(value.values()) for value in values.values()])

		return Y_, max(error1, error2), elapsed_time

	def plot(self):
		X = self.X
		Y, error, elapsed_time = self.report()
		if self.D3:
			fig = plt.figure(figsize=(8, 6))
			ax = Axes3D(fig, elev=-150, azim=110)
			ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y)
			ax.set_title("Iris DataSet" + self.subtitle(error, elapsed_time))
			ax.set_xlabel("1st eigenvector")
			ax.w_xaxis.set_ticklabels([])
			ax.set_ylabel("2nd eigenvector")
			ax.w_yaxis.set_ticklabels([])
			ax.set_zlabel("3rd eigenvector")
			ax.w_zaxis.set_ticklabels([])
		else:
			x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
			y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

			fig = plt.figure(figsize=(8, 6))
			ax = fig.add_subplot(1,1,1)

			ax.scatter(X[:, 0], X[:, 1], c=Y)
			ax.set_xlabel('Sepal length')
			ax.set_ylabel('Sepal width')

			ax.set_title("Iris DataSet" + self.subtitle(error, elapsed_time))

	def subtitle(self, error, elapsed_time):
		if error == 0 and elapsed_time == 0:
			return ""
		else:
			return (" - %s (Error %d/150).\nExecution time: %f [s]" % (self.TITLE, error, elapsed_time) if self.TITLE else "")

	def testing(self):
		a = [[(key, elem) for elem in self.testing_params[key]] for key in self.testing_params.keys()]
		for params in [{elem2[0]: elem2[1] for elem2 in elem} for elem in reduce(conv, a)]:
			self.params = params
			print params, "Error: %d. Tiempo: %f" % (self.report())[1:]


class kmeans(Cluster):
	TITLE = "K-Means"

	def __init__(self, X, Y):
		Cluster.__init__(self, X, Y)

		self.testing_params = {
			"n_clusters": [3],
			"n_init": [2,3,4],
			"init": ['k-means++'],
			"max_iter": [300], #range(200,500),
			"tol": map(lambda x: 10**-x, range(1,10))+[0,],
			"n_jobs": [1]
		}

	def run(self):
		k_means = KMeans(**self.params)
		k_means.fit(self.X)
		return k_means.labels_


class meanshift(Cluster):
	TITLE = "Mini Batch K-Means"

	def __init__(self, X, Y):
		Cluster.__init__(self, X, Y)
		self.params = {"n_components" : CLUSTERS}

	def run(self):
		X = manifold.SpectralEmbedding(**self.params).fit_transform(self.X)
		k_means = KMeans(n_clusters=CLUSTERS)
		k_means.fit(self.X)
		return k_means.labels_

class hac(Cluster):
	TITLE = "HAC"

	def __init__(self, X, Y):
		Cluster.__init__(self, X, Y)
		self.params = {"n_clusters" : CLUSTERS, "linkage" : "complete"}

	def run(self):
		clustering = AgglomerativeClustering(**self.params)
		clustering.fit(self.X)
		return clustering.labels_

class minibatch(Cluster):
	TITLE = "Mini Batch K-Means"

	def __init__(self, X, Y):
		Cluster.__init__(self, X, Y)
		self.params = {"n_clusters" : CLUSTERS, "init": "k-means++"}

	#max_iter=100, batch_size=100, verbose=0, tol=0.0, max_no_improvement=10, n_init=3, reassignment_ratio=0.01
	def run(self):
		clustering = sklearn.cluster.MiniBatchKMeans(**self.params)
		clustering.fit(self.X)
		return clustering.labels_

class ward(Cluster):
	TITLE = "Ward"
	def run(self):
		clustering = sklearn.cluster.Ward(**self.params)
		clustering.fit(self.X)
		return clustering.labels_

class dbscan(Cluster):
	TITLE = "DBScan"

	def __init__(self, X, Y):
		Cluster.__init__(self, X, Y)
		self.params = {"min_samples" : CLUSTERS}

	def run(self):
		clustering = sklearn.cluster.DBSCAN(**self.params)
		clustering.fit(self.X)
		return clustering.labels_
	
def spectral(X):
	return spectral_clustering(X, n_clusters=CLUSTERS, n_components=4,
	eigen_solver='arpack', assign_labels='kmeans', n_init=1)

class cmeans(Cluster):
	TITLE = "Fuzzy CMeans"

	def __init__(self, X, Y):
		Cluster.__init__(self, X, Y)
		self.params = {"c": CLUSTERS, "m": 2, "error": None, "maxiter": 10, "seed": None}

	def run(self):
		X = manifold.SpectralEmbedding(n_components=CLUSTERS).fit_transform(self.X)
		cntr, U, U0, d, Jm, p, fpc = skfuzzy.cmeans(X.T, **self.params)
		UT = U.T
		return map(lambda a: a.index(max(a)), [UT[i,].tolist() for i in range(len(UT))])


iris = datasets.load_iris()

X = iris.data
Y = iris.target

kmeans(X,Y).testing()


Cluster(X,Y).plot()

kmeans(X,Y).plot()
meanshift(X,Y).plot()
minibatch(X,Y).plot()
ward(X,Y).plot()
dbscan(X,Y).plot()
hac(X,Y).plot()
cmeans(X,Y).plot()

plt.show()