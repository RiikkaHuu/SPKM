import numpy as np


"""
This file contains kernels in a format compatible with the Sparse Pre-image Kernel Machine implementation, including
their gradients.
RBF and Polynomial kernels are implemented.
Any new kernels one wants to introduce to SPKM should follow the structure imposed in the class Kernel.
All kernels should include the indicator "zeroone"; if their values are in [0,1]. If True, the SPKM implementation will
automatically rescale the range to [-1, 1] (required with cosine proximity loss in order to it to work.) 

@ Riikka Huusari 2021
"""


class Kernel:

    def __init__(self, params):
        """
        Initialises the kernel function with its parameters
        :param params: type depends on kernel; for polynomial a dictionary, for RBF a float
        """
        self.params = params

    def _kernel(self, X, u):
        """
        Evaluates the kernel function on X and u
        :param X: Data samples, assumed to being on rows
        :param u: should be one-dimensional numpy array
        :return: vector of kernel evaluations
        """
        pass

    def kernel(self, X, u):
        """
        Evaluates the kernel function on X and U (two sets of samples)
        :param X: Data samples, assumed to being on rows
        :param u: Vector, if 2d numpy array should be row vector also
        :return: vector of kernel evaluations
        """

        u = np.squeeze(u)

        if u.ndim == 1:
            return self._kernel(X, u)
        else:
            assert X.shape[1] == u.shape[1]
            ans = np.zeros((X.shape[0], u.shape[0]))
            for ii in range(u.shape[0]):
                ans[:, ii] = self._kernel(X, np.squeeze(u[ii, :]))
            return ans

    def gradient(self, X, u):
        """
        Calculates the gradient of the kernel function evaluated as k(X,u) w.r.t. u
        :param X: Data samples, assumed to being on rows
        :param u: Vector, if 2d numpy array should be row vector also
        :return: a matrix; gradients for every x in X w.r.t u on rows
        """
        pass


class PolyKernel(Kernel):

    def __init__(self, params):
        super().__init__(params)
        assert isinstance(params, dict)
        assert "r" in params
        assert "d" in params
        self.zeroone = False

    def _kernel(self, X, u):
        r = self.params["r"]
        d = self.params["d"]

        return (np.dot(X, u.T) + r * np.ones(X.shape[0])) ** d

    def gradient(self, X, u):
        r = self.params["r"]
        d = self.params["d"]
        t0 = r + np.dot(X, u.T)
        t1 = np.power(t0, d - 1)
        return d * X * t1[:, np.newaxis]


class PolyNormalizedKernel(Kernel):

    def __init__(self, params):
        super().__init__(params)
        assert isinstance(params, dict)
        assert "r" in params
        assert "d" in params

        self.zeroone = False
        if params["d"] % 2 == 0:
            self.zeroone = True

    def _kernel(self, X, u):
        r = self.params["r"]
        d = self.params["d"]

        polyk = (np.dot(X, u.T) + r * np.ones(X.shape[0])) ** d
        kuu = (np.dot(u, u.T) + r) ** d
        kww = ((X * X).sum(-1) + r * np.ones(X.shape[0])) ** d  # diagonal elements of dot products

        assert 0 not in kww
        assert 0 != kuu
        polyk = polyk / np.sqrt(kww * kuu)

        return polyk

    def gradient(self, X, u):

        r = self.params["r"]
        d = self.params["d"]

        tmpK = PolyKernel(self.params)
        gx = tmpK.kernel(X, u)
        dgx = tmpK.gradient(X, u)

        kuu = (np.dot(u, u.T) + r) ** d
        kww = ((X * X).sum(-1) + r * np.ones(X.shape[0])) ** d
        hx2 = kuu * kww
        hx = np.sqrt(kuu * kww)
        dkuu = 2 * d * (np.dot(u, u.T) + r) ** (d - 1) * u
        dhx = dkuu * (0.5 * (kuu * kww) ** (-0.5) * kww)[:, np.newaxis]

        # derivative with quotient rule
        return (dgx * hx[:, np.newaxis] - dhx * gx[:, np.newaxis]) / hx2[:, np.newaxis]


class RBFKernel(Kernel):

    def __init__(self, params):
        super().__init__(params)
        assert isinstance(params, float) or isinstance(params, int)

        self.zeroone = True

    def _kernel(self, X, u):
        sigma = self.params

        xx = (X * X).sum(-1)  # diagonal of WW^T
        yy = np.dot(np.squeeze(u), np.squeeze(u))
        xy = 2 * np.dot(X, np.squeeze(u))  # note 2 is here!

        return np.exp((-xx + xy - yy) / (2 * sigma ** 2))

    def gradient(self, X, u):

        s = self.params

        k = self.kernel(X, u)
        gradients = k[:, np.newaxis] * (-1 / (s ** 2)) * (u[np.newaxis, :] - X)

        return gradients


class TanimotoKernel(Kernel):

    def __init__(self, params):
        super().__init__(params)
        assert isinstance(params, float) or isinstance(params, int)

        self.zeroone = True

    def _kernel(self, X, u):
        tmp = np.dot(X, u)
        ans = tmp / ((X * X).sum(-1) + np.dot(u, u) - tmp)
        return ans

    def gradient(self, X, u):
        # with quotient rule
        h = ((X * X).sum(-1) + np.dot(u, u.T) - np.dot(X, u.T))
        return (h[:, np.newaxis] * X - np.dot(X, u.T)[:, np.newaxis] * (1 - X)) / h[:, np.newaxis] ** 2
