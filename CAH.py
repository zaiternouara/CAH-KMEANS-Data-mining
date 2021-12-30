# importation des bibliothèques
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import DistanceMetric
# mettre le jeu de données dans un tableau
tab = np.array([[1.95,0.97],
    [1.62,0.74],
    [3.12 ,1.85],
    [0.91,1.09],
    [2.37,4.11],
    [5.2,2.52],
    [5.74,5.04],
    [3.00,3.47],
    [4.70,3.65],
    [4.97,3.32],])
# présentation de l'ensemble des points en nuages de points
labels = range(1, 11)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(tab[:,0],tab[:,1], label='True Position')

for label, x, y in zip(labels, tab[:, 0], tab[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
plt.show()
# présentation de la hiérarchie en utilisant le single linkage
linked = linkage(tab, 'single')

labelList = range(1, 11)

plt.figure(figsize=(10, 7))
# le dendrogramme du single linkage
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()
dis = DistanceMetric.get_metric('euclidean')
sing= dis.pairwise(tab)
print(sing)
# présentation de la hiérarchie en utilisant le complete linkage
linked = linkage(tab, 'complete')
labelList = range(1, 11)
plt.figure(figsize=(10, 7))
# le dendrogramme du complete linkage
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()
dist = DistanceMetric.get_metric('euclidean')
comp= dist.pairwise(tab)
print(comp)
# présentation de la hiérarchie en utilisant le average linkage
linked = linkage(tab, 'average')
labelList = range(1, 11)
plt.figure(figsize=(10, 7))
# le dendrogramme du average linkage
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()
dist = DistanceMetric.get_metric('euclidean')
avg= dist.pairwise(tab)
print(avg)


