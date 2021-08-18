import numpy as np
import os

from sklearn.svm import SVC
from spkm import SPKM as Algo
from sklearn.datasets import make_gaussian_quantiles

from kernels_and_gradients_old import RBFKernel
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors


# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ some parameters for plotting ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤

imgfolder = "fig_spkm_cl_illustration/"
if not os.path.exists(imgfolder):
    os.makedirs(imgfolder)

fsize=20
cNorm = colors.Normalize(vmin=-2, vmax=2)  # normalise the colormap
scalarMap = cm.ScalarMappable(norm=cNorm, cmap='RdBu')  # map numbers to colors
# scalarMap2 = cm.ScalarMappable(norm=cNorm, cmap='RdBu')  # map numbers to colors
lightshade = 0.75
darkshade = 1.5
fs = 3
markersize=60
cross_markersize = 80

# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ the data ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤

np.random.seed(112)

X1, Y1 = make_gaussian_quantiles(n_features=2, n_classes=2)
Y1[Y1==0] = -1

ntr = 50
order = np.random.permutation(100)
training = order[:ntr]
test = order[ntr:]

plt.figure(figsize=(fs, fs))
# plt.scatter(X1[:, 0], X1[:, 1], c=scalarMap.to_rgba(Y1))
plt.scatter(X1[training, 0], X1[training, 1], c=scalarMap.to_rgba(Y1[training]*darkshade), edgecolor='k', marker='D', s=markersize)
plt.scatter(X1[test, 0], X1[test, 1], c=scalarMap.to_rgba(Y1[test]*lightshade), edgecolor='k', s=markersize)
plt.title("Dataset", fontsize=fsize)
plt.xticks([])
plt.yticks([])
plt.axis("equal")
plt.tight_layout()
plt.savefig(imgfolder+"dataset")

# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ SVM ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤

svm = SVC(kernel="rbf", gamma=(1/(2*(np.mean(pairwise_distances(X1[training, :]))/2)**2)))
svm.fit(X1[training, :], Y1[training])
svmpreds = svm.predict(X1[test, :])
allsvmpreds = svm.predict(X1)
svmcoefs = np.squeeze(svm.dual_coef_)
svs = svm.support_vectors_
bias = svm.intercept_

print("number of support vectors in SVM:", len(svmcoefs))

plt.figure(figsize=(fs, fs))
plt.scatter(X1[test, 0], X1[test, 1], c=scalarMap.to_rgba(svmpreds*lightshade), edgecolor='k', s=markersize)
# plt.scatter(svs[:, 0], svs[:, 1], marker='x', c=scalarMap.to_rgba(np.sign(svmcoefs)*darkshade), s=np.abs(svmcoefs)*40)
pinds = np.where(svmcoefs>0)[0]
ninds = np.where(svmcoefs<0)[0]
plt.scatter(svs[pinds, 0], svs[pinds, 1], marker='x',
            c=[scalarMap.to_rgba(darkshade)], s=np.abs(svmcoefs[pinds])*cross_markersize)
plt.scatter(svs[ninds, 0], svs[ninds, 1], marker='+',
            c=[scalarMap.to_rgba(-darkshade)], s=np.abs(svmcoefs[ninds])*cross_markersize)
plt.title("SVM - %2d"%(100*accuracy_score(Y1[test], svmpreds))+"%", fontsize=fsize)
plt.xticks([])
plt.yticks([])
plt.axis("equal")
plt.tight_layout()
plt.savefig(imgfolder+"svm")

# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ KRR ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤

kernel = RBFKernel(np.mean(pairwise_distances(X1[training, :]))/2)
Ktr = kernel.kernel(X1[training, :], X1[training, :])
Kts = kernel.kernel(X1[test, :], X1[training, :])
Kall = kernel.kernel(X1, X1[training, :])
c = np.dot(np.linalg.pinv(Ktr+0.001*np.eye(Ktr.shape[0])), Y1[training])
krrpreds = np.sign(np.dot(Kts, c))
krrpreds_all = np.sign(np.dot(Kall, c))

largestc = np.max(np.abs(c))
markerscale = (2*cross_markersize)/largestc

plt.figure(figsize=(fs, fs))
plt.scatter(X1[test, 0], X1[test, 1], c=scalarMap.to_rgba(krrpreds*lightshade), edgecolor='k', s=markersize)
# plt.scatter(X1[training, 0], X1[training, 1], marker='x', c=scalarMap.to_rgba(np.sign(c)*darkshade), s=np.abs(c))
pinds = np.where(c>0)[0]
ninds = np.where(c<0)[0]
plt.scatter(X1[training, 0][pinds], X1[training, 1][pinds], marker='x',
            c=[scalarMap.to_rgba(darkshade)], s=np.abs(c[pinds])*markerscale)
plt.scatter(X1[training, 0][ninds], X1[training, 1][ninds], marker='+',
            c=[scalarMap.to_rgba(-darkshade)], s=np.abs(c[ninds])*markerscale)
plt.title("KRR - %2d"%(100*accuracy_score(Y1[test], krrpreds))+"%", fontsize=fsize)
plt.xticks([])
plt.yticks([])
plt.axis("equal")
plt.tight_layout()
plt.savefig(imgfolder+"krr")

# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ parameters for spkm ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤

# kernel = RBFKernel(np.mean(pairwise_distances(X1[training, :]))/2)  # same kernel as before
algo = Algo()

n_u = 10
# cvec = [-1]  # optionally give cvec if want to try some other initialisation than the default

# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ run spkm ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤

u, cvec = algo.train(X1[training, :], kernel, Y1[training], n_u, 0, 0,
                     classification=True, closs="sq", creg=1, stepsize=1,
                     standardise=False)  #, c=cvec)
preds = algo.predict(X1[test, :])
preds = np.sign(preds)
print("number of large multipliers in algorithm:", len(cvec[np.abs(cvec)>1e-3]))
print("accuracy:", accuracy_score(Y1[test], np.sign(preds)))

largestc = np.max(np.abs(cvec))
markerscale = (2*cross_markersize)/largestc

plt.figure(figsize=(fs, fs))
plt.scatter(X1[test, 0], X1[test, 1], c=scalarMap.to_rgba(preds*lightshade), edgecolor='k', s=markersize)
# plt.scatter(u[:, 0], u[:, 1], marker='x', c=scalarMap.to_rgba(np.sign(cvec)*darkshade), s=np.abs(cvec)*markerscale)
pinds = np.where(cvec>0)[0]
ninds = np.where(cvec<0)[0]
plt.scatter(u[:, 0][pinds], u[:, 1][pinds], marker='x',
            c=[scalarMap.to_rgba(darkshade)], s=np.abs(cvec[pinds])*markerscale)
plt.scatter(u[:, 0][ninds], u[:, 1][ninds], marker='+',
            c=[scalarMap.to_rgba(-darkshade)], s=np.abs(cvec[ninds])*markerscale)
plt.title("SPKM %d - %2d"%(n_u, 100*accuracy_score(Y1[test], preds))+"%", fontsize=fsize)
plt.xticks([])
plt.yticks([])
plt.axis("equal")
plt.tight_layout()
plt.savefig(imgfolder+"spkm_"+str(n_u)+"u")
