import numpy as np


"""
This file contains loss functions in a format compatible with the Sparse Pre-image Kernel Machine implementation, 
including their gradients.
Cosine proximity and squared losses are implemented.
Any new losses one wants to introduce to SPKM should follow the structure imposed in the class Loss.

@ Riikka Huusari 2021
"""


# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
# baseclass

class Loss:

    def __init__(self, X, kernel, Y):
        self.X = X
        self.kernel = kernel
        self.Y = Y

    def cost_uc(self, u, c):
        pass

    def grad_u(self, u, c):
        pass


# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
# binary classification, regression

class SqLoss(Loss):

    def cost_uc(self, u, c):

        u = np.squeeze(u)
        if self.kernel.zeroone:
            ku = self.kernel.kernel(self.X, u) * 2 - 1
        else:
            ku = self.kernel.kernel(self.X, u)
        if len(c) == 1:
            kuc = ku * c
        else:
            kuc = np.dot(ku, c)

        return np.linalg.norm(kuc - self.Y) ** 2

    def grad_u(self, u, c):

        if u.ndim == 2:
            n_u = u.shape[0]
        else:
            n_u = len(u)

        if self.kernel.zeroone:
            ku = self.kernel.kernel(self.X, u) * 2 - 1
        else:
            ku = self.kernel.kernel(self.X, u)

        if len(c) == 1:
            kuc = ku * c
        else:
            kuc = np.dot(ku, c)

        gku_sum = np.zeros(self.X.shape)
        for ii in range(n_u):
            if n_u == 1:
                utmp = np.squeeze(u)
            else:
                utmp = np.squeeze(u[ii, :])
            if self.kernel.zeroone:
                gku_sum += 2 * self.kernel.gradient(self.X, utmp)
            else:
                gku_sum += self.kernel.gradient(self.X, utmp)

        return np.outer(2 * c, np.dot((kuc - self.Y).T, gku_sum))


class CosLoss(Loss):

    def cost_uc(self, u, c):
        u = np.squeeze(u)
        if self.kernel.zeroone:
            ku = self.kernel.kernel(self.X, u) * 2 - 1
        else:
            ku = self.kernel.kernel(self.X, u)
        if len(c) == 1:
            kuc = ku * c
        else:
            kuc = np.dot(ku, c)
        return -np.dot(c, np.dot(ku.T, self.Y)) / (np.linalg.norm(kuc))

    def grad_u(self, u, c):

        if u.ndim == 2:
            n_u = u.shape[0]
        else:
            n_u = len(u)

        if self.kernel.zeroone:
            ku = self.kernel.kernel(self.X, u) * 2 - 1
        else:
            ku = self.kernel.kernel(self.X, u)

        if len(c) == 1:
            kuc = np.squeeze(ku * c)
        else:
            kuc = np.dot(ku, c)

        gku_sum = np.zeros(self.X.shape)
        for ii in range(n_u):
            if n_u == 1:
                utmp = np.squeeze(u)
            else:
                utmp = np.squeeze(u[ii, :])
            if self.kernel.zeroone:
                gku_sum += 2 * self.kernel.gradient(self.X, utmp)
            else:
                gku_sum += self.kernel.gradient(self.X, utmp)

        # naming convention like in wikipedia for quotient rule
        hx = np.sqrt(np.dot(kuc, kuc))
        gx = np.dot(kuc.T, self.Y)
        dgx = np.outer(c, np.dot(self.Y.T, gku_sum))
        dhx = (np.outer(np.dot(gku_sum.T, kuc), c)) / (hx)

        # print(dgx.shape, hx.shape, gx.shape, dhx.shape)

        grad = (dgx * hx - gx * dhx.T) / (hx ** 2)
        return -grad

    def grad_c(self, c, u):

        u = np.squeeze(u)

        if u.ndim == 2:
            n_u = u.shape[0]
        else:
            n_u = len(u)

        if self.kernel.zeroone:
            ku = self.kernel.kernel(self.X, u) * 2 - 1
        else:
            ku = self.kernel.kernel(self.X, u)

        if np.squeeze(u.ndim) == 1:
            kuc = ku * c
        else:
            kuc = np.dot(ku, c)

        gc = np.dot(kuc.T, self.Y)
        hc = np.sqrt(np.dot(kuc.T, kuc))
        B = np.dot(ku.T, ku)
        dhc = (hc**(-0.5))*np.dot(B, c)
        dgc = np.dot(ku.T, self.Y)

        grad = (dgc * hc - gc * dhc.T) / (hc ** 2)
        return -np.squeeze(grad)


class LogLoss(Loss):

    # todo check this!! not used since ages

    def cost_u(self, u, c):

        u = np.squeeze(u)
        if self.kernel.zeroone:
            ku = self.kernel.kernel(self.X, u) * 2 - 1
        else:
            ku = self.kernel.kernel(self.X, u)

        fx = np.dot(ku, c)

        return np.sum(np.log(1 + np.exp(-np.multiply(self.Y, fx))))

    def grad_u(self, u, c):

        if u.ndim == 2:
            n_u = u.shape[0]
        else:
            n_u = len(u)

        u = np.squeeze(u)
        if self.kernel.zeroone:
            ku = self.kernel.kernel(self.X, u) * 2 - 1
        else:
            ku = self.kernel.kernel(self.X, u)

        fx = np.dot(ku, c)

        upstairs = np.exp(-np.multiply(self.Y, fx))
        downstairs = 1 + upstairs
        upstairs = np.multiply(upstairs, self.Y)

        multipliers = (-upstairs / downstairs)[:, np.newaxis]

        dku = np.zeros(self.X.shape)
        for ii in range(n_u):
            if self.kernel.zeroone:
                gu = self.kernel.gradient(self.X, u[ii, :]) * 2
            else:
                gu = self.kernel.gradient(self.X, u[ii, :])
            gu = c[ii] * gu
            # now gu is of shape n*d; need to sum over the n
            # but in derivative also multiply with the upstairs/downstairs
            dku[ii, :] = np.sum(np.multiply(multipliers, gu), axis=0)

        return dku


# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
# multi-class

class MCCosLoss(Loss):

    # for this multi-class loss Y should be given one-hot encoded

    def cost_uc(self, u, c):
        nc = self.Y.shape[1]
        u = np.squeeze(u)
        if self.kernel.zeroone:
            ku = self.kernel.kernel(self.X, u) * 2 - 1
        else:
            ku = self.kernel.kernel(self.X, u)
        res = 0
        for nnc in range(nc):
            kuc = np.dot(ku, c[:, nnc])
            res -= np.dot(c[:, nnc], np.dot(ku.T, self.Y[:, nnc])) / (
                np.linalg.norm(kuc))
        return res

    def grad_u(self, u, c):

        nc = self.Y.shape[1]
        n_u = u.shape[0]

        # sum of individual cosine losses!
        res = 0

        if self.kernel.zeroone:
            ku = self.kernel.kernel(self.X, u) * 2 - 1
        else:
            ku = self.kernel.kernel(self.X, u)

        gku_sum = np.zeros(self.X.shape)
        for ii in range(n_u * nc):
            utmp = np.squeeze(u[ii, :])
            if self.kernel.zeroone:
                gku_sum += 2 * self.kernel.gradient(self.X, utmp)  # I assume the derivatives are on rows
            else:
                gku_sum += self.kernel.gradient(self.X, utmp)  # I assume the derivatives are on rows

        for nnc in range(nc):
            kuc = np.dot(ku, c[:, nnc])

            # naming convention like in wikipedia for quotient rule

            hx = np.sqrt(np.dot(kuc, kuc))
            gx = np.dot(kuc.T, self.Y[:, nnc])
            dgx = np.outer(c[:, nnc], np.dot(self.Y[:, nnc].T, gku_sum))
            dhx = (np.outer(np.dot(gku_sum.T, kuc), c[:, nnc])) / (hx)

            # print(dgx.shape, hx.shape, gx.shape, dhx.shape)

            grad = (dgx * hx - gx * dhx.T) / (hx ** 2)
            res -= grad

        return res


# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
# multi-view (MKL style)

# todo this
class MVCosLoss(Loss):

    def cost_uc(self, u, c):

        n = self.X.shape[0]
        nn_us = len(c)

        kuall = np.zeros((n, nn_us))
        indx = 0
        for vv in u.keys():

            n_u = 1
            if np.squeeze(u[vv]).ndim == 2:
                n_u = u[vv].shape[0]

            utmp = np.squeeze(u[vv])
            if self.kernel[vv].zeroone:
                ku = self.kernel[vv].kernel(self.X[vv], utmp) * 2 - 1
            else:
                ku = self.kernel[vv].kernel(self.X[vv], utmp)
            kuall[:, indx:(indx + n_u)] = ku
            indx += n_u

        return -np.dot(c, np.dot(kuall.T, self.Y)) / (
            np.linalg.norm(np.dot(kuall, c)))

    def grad_u(self, u, c):  # returns a dictionary!!!!!!!!!!!!

        n = self.X.shape[0]
        nn_us = len(c)

        kuall = np.zeros((n, nn_us))
        kucs = {}
        indx = 0
        cindx = 0
        for vv in u.keys():
            n_u = 1
            if np.squeeze(u[vv]).ndim == 2:
                n_u = u[vv].shape[0]
            utmp = np.squeeze(u[vv])
            if self.kernel[vv].zeroone:
                ku = self.kernel[vv].kernel(self.X[vv], utmp) * 2 - 1
            else:
                ku = self.kernel[vv].kernel(self.X[vv], utmp)
            kuall[:, indx:(indx + n_u)] = ku
            indx += n_u
            kucs[vv] = np.dot(ku, c[cindx:(cindx+n_u)])
            cindx += n_u

        kuc = np.dot(kuall, c)

        gku_sum = {}
        for vv in u.keys():
            gku_sum[vv] = np.zeros((n, self.X[vv].shape[1]))
            n_u = 1
            if np.squeeze(u[vv]).ndim == 2:
                n_u = u[vv].shape[0]
            for ii in range(n_u):
                if self.kernel[vv].zeroone:
                    # todo why do I get overflow in gradient?
                    # print(u[vv][ii, :])
                    # print(kernels[vv].gradient(X[vv], np.squeeze(u[vv][ii, :])))
                    gku_sum[vv] += 2 * self.kernel[vv].gradient(self.X[vv], np.squeeze(u[vv][ii, :]))
                else:
                    gku_sum[vv] += self.kernel[vv].gradient(self.X[vv], np.squeeze(u[vv][ii, :]))

        # naming convention like in wikipedia for quotient rule

        # these are just scalars... the same for all of these?
        hx = np.sqrt(np.dot(kuc, kuc))
        gx = np.dot(kuc.T, self.Y)
        # and if I just take these gradients one-by-one?

        grad = {}

        cindx = 0
        for vv in u.keys():
            n_u = 1
            if np.squeeze(u[vv]).ndim == 2:
                n_u = u[vv].shape[0]
            dgx = np.outer(c[cindx:(cindx+n_u)], np.dot(self.Y.T, gku_sum[vv]))
            dhx = (np.outer(np.dot(gku_sum[vv].T, kucs[vv]), c[cindx:(cindx+n_u)])) / (hx)

            # print(dgx.shape, hx.shape, gx.shape, dhx.shape)

            grad[vv] = - (dgx * hx - gx * dhx.T) / (
                    hx ** 2)  # note: minus here as I want to maximize!!!!!

        return grad


# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
# multi-view (pairwise)

class PairwiseCosLoss(Loss):

    def __init__(self, X, kernel, Y):
        self.X = X[0]
        self.Z = X[2]
        self.xinds = X[1]
        self.zinds = X[3]
        self.kx = kernel[0]
        self.kz = kernel[1]
        self.Y = Y

    def cost_uvc(self, u, v, c, elems=-1):

        if elems == -1:
            # all of them
            elems = np.arange(len(self.Y))

        u = np.squeeze(u)
        if self.kx.zeroone:
            ku = self.kx.kernel(self.X[self.xinds[elems], :], u) * 2 - 1
        else:
            ku = self.kx.kernel(self.X[self.xinds[elems], :], u)
        v = np.squeeze(v)
        if self.kz.zeroone:
            kv = self.kz.kernel(self.Z[self.zinds[elems], :], v) * 2 - 1
        else:
            kv = self.kz.kernel(self.Z[self.zinds[elems], :], v)
        kvec = ku*kv
        return -np.dot(c, np.dot(kvec.T, self.Y[elems])) / (np.linalg.norm(np.dot(kvec, c)))

    def grad_u(self, u, v, c, elems=-1):

        if elems == -1:
            elems = np.arange(len(self.Y))

        if self.kx.zeroone:
            ku = self.kx.kernel(self.X[self.xinds[elems], :], u) * 2 - 1
        else:
            ku = self.kx.kernel(self.X[self.xinds[elems], :], u)
        if self.kz.zeroone:
            kv = self.kz.kernel(self.Z[self.zinds[elems], :], v) * 2 - 1
        else:
            kv = self.kz.kernel(self.Z[self.zinds[elems], :], v)

        kuc = np.dot(ku*kv, c)

        gku_sum = np.zeros((len(elems), self.X.shape[1]))
        for ii in range(len(c)):
            if self.kx.zeroone:
                gku_sum += 2 * np.squeeze(kv[:, ii])[:, np.newaxis] * \
                           self.kx.gradient(self.X[self.xinds[elems], :], np.squeeze(u[ii, :]))
            else:
                gku_sum += np.squeeze(kv[:, ii])[:, np.newaxis] * \
                           self.kx.gradient(self.X[self.xinds[elems], :], np.squeeze(u[ii, :]))
        gku_sum = gku_sum

        # naming convention like in wikipedia for quotient rule

        hx = np.sqrt(np.dot(kuc, kuc))
        gx = np.dot(kuc.T, self.Y[elems])
        dgx = np.outer(c, np.dot(self.Y[elems].T, gku_sum))
        dhx = (np.outer(np.dot(gku_sum.T, kuc), c)) / (hx)

        # print(dgx.shape, hx.shape, gx.shape, dhx.shape)

        grad = (dgx * hx - gx * dhx.T) / (hx ** 2)

        return -grad

    def grad_v(self, v, u, c, elems=-1):

        if elems == -1:
            elems = np.arange(len(self.Y))

        if self.kx.zeroone:
            ku = self.kx.kernel(self.X[self.xinds[elems], :], u) * 2 - 1
        else:
            ku = self.kx.kernel(self.X[self.xinds[elems], :], u)
        if self.kz.zeroone:
            kv = self.kz.kernel(self.Z[self.zinds[elems], :], v) * 2 - 1
        else:
            kv = self.kz.kernel(self.Z[self.zinds[elems], :], v)

        kuc = np.dot(ku*kv, c)

        gkv_sum = np.zeros((len(elems), self.Z.shape[1]))
        for ii in range(len(c)):
            if self.kz.zeroone:
                gkv_sum += 2 * np.squeeze(ku[:, ii])[:, np.newaxis] * \
                           self.kz.gradient(self.Z[self.zinds[elems], :], np.squeeze(v[ii, :]))
            else:
                gkv_sum += np.squeeze(ku[:, ii])[:, np.newaxis] * \
                           self.kz.gradient(self.Z[self.zinds[elems], :], np.squeeze(v[ii, :]))
        gkv_sum = gkv_sum

        # naming convention like in wikipedia for quotient rule

        hx = np.sqrt(np.dot(kuc, kuc))
        gx = np.dot(kuc.T, self.Y[elems])
        dgx = np.outer(c, np.dot(self.Y[elems].T, gkv_sum))
        dhx = (np.outer(np.dot(gkv_sum.T, kuc), c)) / (hx)

        # print(dgx.shape, hx.shape, gx.shape, dhx.shape)

        grad = (dgx * hx - gx * dhx.T) / (hx ** 2)

        return -grad

    def grad_c(self, c, u, v):

        u = np.squeeze(u)

        if self.kx.zeroone:
            ku = self.kx.kernel(self.X[self.xinds, :], u) * 2 - 1
        else:
            ku = self.kx.kernel(self.X[self.xinds, :], u)
        if self.kz.zeroone:
            kv = self.kz.kernel(self.Z[self.zinds, :], v) * 2 - 1
        else:
            kv = self.kz.kernel(self.Z[self.zinds, :], v)

        kuc = np.dot(ku * kv, c)

        gc = np.dot(kuc.T, self.Y)
        hc = np.sqrt(np.dot(kuc.T, kuc))
        B = np.dot(ku.T, ku)
        dhc = (hc**(-0.5))*np.dot(B, c)
        dgc = np.dot(ku.T, self.Y)

        grad = (dgc * hc - gc * dhc.T) / (hc ** 2)
        return -np.squeeze(grad)


class PairwiseSqLoss(Loss):

    def __init__(self, X, kernel, Y):
        self.X = X[0]
        self.Z = X[2]
        self.xinds = X[1]
        self.zinds = X[3]
        self.kx = kernel[0]
        self.kz = kernel[1]
        self.Y = Y

    def cost_uvc(self, u, v, c, elems=None):

        if elems is None:
            # all of them
            elems = np.arange(len(self.Y))

        u = np.squeeze(u)
        if self.kx.zeroone:
            ku = self.kx.kernel(self.X[self.xinds[elems], :], u) * 2 - 1
        else:
            ku = self.kx.kernel(self.X[self.xinds[elems], :], u)
        v = np.squeeze(v)
        if self.kz.zeroone:
            kv = self.kz.kernel(self.Z[self.zinds[elems], :], v) * 2 - 1
        else:
            kv = self.kz.kernel(self.Z[self.zinds[elems], :], v)

        if len(c) == 1:
            kuvc = (ku*kv)*c
        else:
            kuvc = np.dot(ku*kv, c)
        return np.linalg.norm(kuvc - self.Y[elems]) ** 2

    def grad_u(self, u, v, c, elems=None):

        if elems is None:
            elems = np.arange(len(self.Y))

        if self.kx.zeroone:
            ku = self.kx.kernel(self.X[self.xinds[elems], :], u) * 2 - 1
        else:
            ku = self.kx.kernel(self.X[self.xinds[elems], :], u)
        if self.kz.zeroone:
            kv = self.kz.kernel(self.Z[self.zinds[elems], :], v) * 2 - 1
        else:
            kv = self.kz.kernel(self.Z[self.zinds[elems], :], v)

        kuc = np.dot(ku*kv, c)

        gku_sum = np.zeros((len(elems), self.X.shape[1]))
        for ii in range(len(c)):
            if self.kx.zeroone:
                gku_sum += 2 * np.squeeze(kv[:, ii])[:, np.newaxis] * \
                           self.kx.gradient(self.X[self.xinds[elems], :], np.squeeze(u[ii, :]))
            else:
                gku_sum += np.squeeze(kv[:, ii])[:, np.newaxis] * \
                           self.kx.gradient(self.X[self.xinds[elems], :], np.squeeze(u[ii, :]))
        gku_sum = gku_sum

        return np.outer(2 * c, np.dot((kuc - self.Y[elems]).T, gku_sum))

    def grad_v(self, v, u, c, elems=None):

        if elems is None:
            elems = np.arange(len(self.Y))

        if self.kx.zeroone:
            ku = self.kx.kernel(self.X[self.xinds[elems], :], u) * 2 - 1
        else:
            ku = self.kx.kernel(self.X[self.xinds[elems], :], u)
        if self.kz.zeroone:
            kv = self.kz.kernel(self.Z[self.zinds[elems], :], v) * 2 - 1
        else:
            kv = self.kz.kernel(self.Z[self.zinds[elems], :], v)

        kuc = np.dot(ku*kv, c)

        gkv_sum = np.zeros((len(elems), self.Z.shape[1]))
        for ii in range(len(c)):
            if self.kz.zeroone:
                gkv_sum += 2 * np.squeeze(ku[:, ii])[:, np.newaxis] * \
                           self.kz.gradient(self.Z[self.zinds[elems], :], np.squeeze(v[ii, :]))
            else:
                gkv_sum += np.squeeze(ku[:, ii])[:, np.newaxis] * \
                           self.kz.gradient(self.Z[self.zinds[elems], :], np.squeeze(v[ii, :]))
        gkv_sum = gkv_sum

        return np.outer(2 * c, np.dot((kuc - self.Y[elems]).T, gkv_sum))

