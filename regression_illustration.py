import numpy as np
import os

from kernels_and_gradients_old import RBFKernel
from spkm import SPKM 
from sklearn.linear_model import Lasso
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances

import warnings

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ¤¤¤¤¤¤¤¤¤¤¤¤¤ some parameters for plotting ¤¤¤¤¤¤¤¤¤¤¤¤¤

imgfolder = "fig_spkm_regr_illustration/"  
if not os.path.exists(imgfolder):
    os.makedirs(imgfolder)

fontsize=20
# cNorm = colors.Normalize(vmin=-2,vmax=2)  # normalise the colormap
# scalarMap = cm.ScalarMappable(norm=cNorm, cmap='RdBu')  # map numbers to colors
# scalarMap2 = cm.ScalarMappable(norm=cNorm, cmap='RdBu')  # map numbers to colors
lightshade = 0.75
midshade = 1
darkshade = 1.5
fs = 3
fsize = (fs, fs)
markersize=60
cross_markersize = 80

vvmin=-1
vvmax=2.0
titlesize = fontsize

cNorm = colors.Normalize(vmin=-2,vmax=2)  # normalise the colormap
scalarMap = cm.ScalarMappable(norm=cNorm, cmap='RdBu')  # map numbers to colors

# ¤¤¤¤¤¤¤¤¤¤¤¤¤ create the dataset ¤¤¤¤¤¤¤¤¤¤¤¤¤

rdataseed = 0
np.random.seed(rdataseed)
n = 200
d = 2
data = np.random.randn(n, d)
f = lambda x: np.sum([x[indx] ** 2 for indx in np.arange(len(x))])
y = []
for ii in range(data.shape[0]):
    y.append(np.sqrt(f(data[ii, :])))
y = np.array(y)

data = (data - np.mean(data)) / np.std(data)
y = (y - np.mean(y)) / np.std(y)

np.random.seed(42)
n = len(y)
order = np.random.permutation(n)
ntr = int(n / 2)
training = order[:ntr]
testing = order[ntr:]

Y = y[training]
Yt = y[testing]
X = data[training, :]
Xt = data[testing, :]

# kernel = RBFKernel(np.mean(pairwise_distances(X))/2)  # much better results with preimage algo than without /2
kernel = RBFKernel(1)  # for standardised data this rbf kernel parameter should be good
Ktr = kernel.kernel(X, X)
Kts = kernel.kernel(Xt, X)

# plot the data

plt.figure(figsize=fsize)
plt.scatter(data[:, 0], data[:, 1], c=y, edgecolor='k', vmin=vvmin, vmax=vvmax, s=markersize)
plt.axis("equal")
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.savefig(imgfolder+"allData")
vmin, vmax = plt.gci().get_clim()
print("colors:", vmin, vmax)

plt.figure(figsize=fsize)
plt.scatter(data[training, 0], data[training, 1], c=y[training], edgecolor='k',
            vmin=vvmin, vmax=vvmax, s=markersize)
plt.title("Training data", fontsize=titlesize)
plt.axis("equal")
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.savefig(imgfolder + "trData")
plt.figure(figsize=fsize)
plt.scatter(data[testing, 0], data[testing, 1], c=y[testing], edgecolor='k',
            vmin=vvmin, vmax=vvmax, s=markersize)
plt.title("Test data", fontsize=titlesize)
plt.axis("equal")
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.savefig(imgfolder + "tsData")

plt.figure(figsize=fsize)
plt.scatter(data[training, 0], data[training, 1], marker="d", c=y[training], edgecolor='k',
            vmin=vvmin, vmax=vvmax, s=markersize)
plt.scatter(data[testing, 0], data[testing, 1], c=y[testing], edgecolor='k',
            vmin=vvmin, vmax=vvmax, s=markersize)
plt.title("Dataset", fontsize=titlesize)
plt.axis("equal")
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.savefig(imgfolder + "trtsData")


# ¤¤¤¤¤¤¤¤¤¤¤¤¤ KRR ¤¤¤¤¤¤¤¤¤¤¤¤¤

lmbda = 0.001
c = np.dot(np.linalg.pinv(Ktr+lmbda*np.eye(Ktr.shape[0])), Y)
krrpreds = np.dot(Kts, c)
print("KRR:", len(c), "training samples")

largestc = np.max(np.abs(c))
markerscale = (2*cross_markersize)/largestc

plt.figure(figsize=(fsize))
plt.scatter(Xt[:, 0], Xt[:, 1], c=krrpreds, edgecolor='k', vmin=vvmin, vmax=vvmax, s=markersize)
# plt.scatter(X[:, 0], X[:, 1], marker='x', c=scalarMap.to_rgba(-np.sign(c)), s=np.abs(c))
pinds = np.where(c<0)[0]
ninds = np.where(c>0)[0]
plt.scatter(X[pinds, 0], X[pinds, 1], marker='x',
            c=[scalarMap.to_rgba(midshade)], s=np.abs(c[pinds])*markerscale)
plt.scatter(X[ninds, 0], X[ninds, 1], marker='+',
            c=[scalarMap.to_rgba(-midshade)], s=np.abs(c[ninds])*markerscale)
plt.title("KRR - %5.3f"%(mean_squared_error(Yt, krrpreds)), fontsize=titlesize)
plt.xticks([])
plt.yticks([])
plt.axis("equal")
plt.tight_layout()
plt.savefig(imgfolder+"krr")

# ¤¤¤¤¤¤¤¤¤¤¤¤¤ SVR ¤¤¤¤¤¤¤¤¤¤¤¤¤

lmbda = 1  # seems ok, smaller are not as great
algo = SVR(kernel="precomputed", C=lmbda)
algo.fit(Ktr, Y)
svrpreds = algo.predict(Kts)
c = np.squeeze(algo.dual_coef_)
cinds = algo.support_
print("SVR:", len(c), "support vectors")

largestc = np.max(np.abs(c))
markerscale = (2*cross_markersize)/largestc

plt.figure(figsize=(fsize))
plt.scatter(Xt[:, 0], Xt[:, 1], c=svrpreds, edgecolor='k', vmin=vvmin, vmax=vvmax, s=markersize)
# plt.scatter(X[cinds, 0], X[cinds, 1], marker='x', c=scalarMap.to_rgba(-np.sign(c)), s=np.abs(c)*100)
pinds = np.where(c<0)[0]
ninds = np.where(c>0)[0]
plt.scatter(X[cinds, 0][pinds], X[cinds, 1][pinds], marker='x',
            c=[scalarMap.to_rgba(midshade)], s=np.abs(c[pinds])*markerscale)
plt.scatter(X[cinds, 0][ninds], X[cinds, 1][ninds], marker='+',
            c=[scalarMap.to_rgba(-midshade)], s=np.abs(c[ninds])*markerscale)
plt.title("SVR - %5.3f"%(mean_squared_error(Yt, svrpreds)), fontsize=titlesize)
plt.xticks([])
plt.yticks([])
plt.axis("equal")
plt.tight_layout()
plt.savefig(imgfolder+"svr")


# ¤¤¤¤¤¤¤¤¤¤¤¤¤ Lasso ¤¤¤¤¤¤¤¤¤¤¤¤¤

# plt.savefig(imgfolder+"krr")
lmbda = 0.000001  # larger predict constant, at least this has some differing values
algo = Lasso(alpha=lmbda)
algo.fit(X, Y)
lassopreds = algo.predict(Xt)
c = algo.coef_

plt.figure(figsize=(fsize))
plt.scatter(Xt[:, 0], Xt[:, 1], c=lassopreds, edgecolor='k', vmin=np.min(c), vmax=np.max(c), s=markersize)
plt.scatter(c[0], c[1], marker='x', c=scalarMap.to_rgba([-1]), s=markersize)
plt.title("Lasso - %5.3f"%(mean_squared_error(Yt, lassopreds)), fontsize=titlesize)
plt.xticks([])
plt.yticks([])
plt.axis("equal")
plt.tight_layout()
plt.savefig(imgfolder+"lasso")

plt.close("all")

# ¤¤¤¤¤¤¤¤¤¤¤¤¤ spkm ¤¤¤¤¤¤¤¤¤¤¤¤¤

# ---- parameters ----
n_u = 5
init = "randn"
innerloops = 20
outerloops = 5

# --------------------

from losses_and_gradients import SqLoss, CosLoss

algo = SPKM()
u, c = algo.train(X, kernel, np.copy(Y), n_u, 0, 0,
                  standardise=True, calculate_new_rbf_param=True,
                  classification=False, loss=CosLoss, init=init, closs="sq", creg=2,
                  max_outer_iters=outerloops, max_gd_iters=innerloops)
preimg_preds = algo.predict(Xt)

sizemultiplier = 150/np.max(np.abs(c))

u = algo.u_unstandardised

largestc = np.max(np.abs(c))
markerscale = (2*cross_markersize)/largestc

# plt.figure(figsize=(fsize))
plt.figure(figsize=fsize)
# plt.subplot(121)
plt.scatter(Xt[:, 0], Xt[:, 1], c=preimg_preds, edgecolor='k', vmin=vvmin, vmax=vvmax, s=markersize)
# plt.scatter(u[:, 0], u[:, 1], marker='x', c=scalarMap.to_rgba(-np.sign(c)), s=np.abs(c)*sizemultiplier)
pinds = np.where(c<0)[0]
ninds = np.where(c>0)[0]
plt.scatter(u[:, 0][pinds], u[:, 1][pinds], marker='x',
            c=[scalarMap.to_rgba(midshade)], s=np.abs(c[pinds])*markerscale)
plt.scatter(u[:, 0][ninds], u[:, 1][ninds], marker='+',
            c=[scalarMap.to_rgba(-midshade)], s=np.abs(c[ninds])*markerscale)
# plt.scatter(0, 0, c="r", edgecolor='k')
plt.title("SPKM "+str(n_u)+" - %5.3f"%(mean_squared_error(Yt, preimg_preds)), fontsize=titlesize)
plt.xticks([])
plt.yticks([])
plt.axis("equal")
plt.tight_layout()
plt.savefig(imgfolder+"preimg"+str(n_u))
