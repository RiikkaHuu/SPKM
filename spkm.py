import numpy as np

from losses_and_gradients import CosLoss, SqLoss, MCCosLoss, MVCosLoss, PairwiseCosLoss

"""
This file contains implementation of the Sparse Pre-image Kernel Machine (SPKM) method. 
Files losses_and_gradients.py and kernels_and_gradients_old.py contain helpers.
Note if implementing new losses: the SPKM preprocessing in some cases assume that classification 
losses work with -1,+1 labels (e.g. the multiclass approach changes labels to one-hot encoding accordingly).


@ Riikka Huusari 2021
"""


# ================================================== helpers ==================================================

def proj_l1(v, b):

    """
    Adapted from https://github.com/aalto-ics-kepaco/gradKCCA
    :param v: this should be projected to the ball
    :param b: probably the P in gradKCCA paper
    :return:
    """

    assert b > 0
    if np.linalg.norm(v, 1) < b:
        return v
    u = -np.sort(-np.abs(v))
    sv = np.cumsum(u)
    # print("u shape:", u.shape, np.arange(u.shape[0]))
    # print((sv - b) / np.arange(1, u.shape[0]+1))
    # print(u)
    # print(np.where(u > (sv - b) / np.arange(1, u.shape[0]+1)))
    r = np.where(u > (sv - b) / np.arange(1, u.shape[0]+1))
    if len(r[-1]) > 0:
        # print("not empty")
        rho = r[-1][-1]
        tau = (sv[rho] - b) / (rho + 1)
        theta = np.maximum(0, tau)
        return np.sign(v) * np.maximum(np.abs(v) - theta, 0)
    else:
        # print("empty")
        return v


# ================================================== SPKM ==================================================

class baseSPKM:

    def __init__(self):
        self.u = None
        self.c = None

    def train(self, X, k, y, n_u, P, gamma, classification=True, loss=CosLoss, c=None, closs="sq", creg=2,
              init="randn", n_inits=5, rseed=0,
              standardise=False, calculate_new_rbf_param=False, standardise_labels=False,
              max_outer_iters=1, max_gd_iters=50, stepsize=10, verbosity=0):

        """

        :param X: data
        :param k: Kernel object
        :param y: labels (assume -1&1 for classification)
        :param n_u: number of u's (basis vectors) to learn
        :param P: parameter for l1-ball projection; 0 if omitted
        :param gamma: parameter for proximal gradient descent; 0 if omitted
        :param loss: Loss object
        :param c: initial c multipliers; if None standard initialisation schemes are used
        :param closs:
        :param init: style of initialisation: "randn", "data", "const"
        :param n_inits: if there is randomness in initialisation, how many times should the algorithm be run
        :param standardise: if data should be standardised (can be useful e.g. in regression)
        :param calculate_new_rbf_param: rbf kernel parameter might need to be changed if data is standardised.
        this will slow down the algorithm; calculating pairwise distances has n^2 scalability
        :param standardise_labels: if labels should be standardised (can be useful in regression)
        :param rseed: random seed for initialisation
        :param max_outer_iters: maximum number of times U and c are updated (1 is often enough for classification)
        :param max_gd_iters: maximum number of iterations for gradient descent
        :param stepsize: initial stepsize for gradient descent
        :param verbosity: if anything should be printed about the algorithm process (0: silent, >0: print stuff)
        :return: U, c
        """

        pass

    def predict(self, Xt):
        pass

    def run_predict(self):
        pass

    def solve_cone_c(self, ku, y):

        """
        Adapted from https://cvxopt.org/examples/book/basispursuit.html
        :param ku:
        :param y:
        :return:
        """

        from cvxopt import matrix, mul, div, sqrt  # cos, sin, exp,
        from cvxopt import blas, lapack, solvers

        A = matrix(ku)
        y_ = matrix(y.astype(float))

        # Basis pursuit problem
        #
        #     minimize    ||A*x - y||_2^2 + ||x||_1
        #
        #     minimize    x'*A'*A*x - 2.0*y'*A*x + 1'*u
        #     subject to  -u <= x <= u
        #
        # Variables x (n),  u (n).

        m, n = A.size
        r = matrix(0.0, (m, 1))

        q = matrix(1.0, (2 * n, 1))

        blas.gemv(A, y_, q, alpha=-2.0, trans='T')

        def P(u, v, alpha=1.0, beta=0.0):
            """
            Function and gradient evaluation of

                v := alpha * 2*A'*A * u + beta * v
            """

            blas.gemv(A, u, r)
            blas.gemv(A, r, v, alpha=2.0 * alpha, beta=beta, trans='T')

        def G(u, v, alpha=1.0, beta=0.0, trans='N'):
            """
                v := alpha*[I, -I; -I, -I] * u + beta * v  (trans = 'N' or 'T')
            """

            blas.scal(beta, v)
            blas.axpy(u, v, n=n, alpha=alpha)
            blas.axpy(u, v, n=n, alpha=-alpha, offsetx=n)
            blas.axpy(u, v, n=n, alpha=-alpha, offsety=n)
            blas.axpy(u, v, n=n, alpha=-alpha, offsetx=n, offsety=n)

        h = matrix(0.0, (2 * n, 1))

        S = matrix(0.0, (m, m))
        Asc = matrix(0.0, (m, n))
        v = matrix(0.0, (m, 1))

        def Fkkt(W):
            # Factor
            #
            #     S = A*D^-1*A' + I
            #
            # where D = 2*D1*D2*(D1+D2)^-1, D1 = d[:n]**2, D2 = d[n:]**2.

            d1, d2 = W['di'][:n] ** 2, W['di'][n:] ** 2

            # ds is square root of diagonal of D
            ds = sqrt(2.0) * div(mul(W['di'][:n], W['di'][n:]), sqrt(d1 + d2))
            d3 = div(d2 - d1, d1 + d2)

            # Asc = A*diag(d)^-1/2
            blas.copy(A, Asc)
            for k in range(m):
                blas.tbsv(ds, Asc, n=n, k=0, ldA=1, incx=m, offsetx=k)

            # S = I + A * D^-1 * A'
            blas.syrk(Asc, S)
            S[::m + 1] += 1.0
            lapack.potrf(S)

            def g(x, y, z):
                x[:n] = 0.5 * (x[:n] - mul(d3, x[n:]) + \
                               mul(d1, z[:n] + mul(d3, z[:n])) - \
                               mul(d2, z[n:] - mul(d3, z[n:])))
                x[:n] = div(x[:n], ds)

                # Solve
                #
                #     S * v = 0.5 * A * D^-1 * ( bx[:n]
                #             - (D2-D1)*(D1+D2)^-1 * bx[n:]
                #             + D1 * ( I + (D2-D1)*(D1+D2)^-1 ) * bz[:n]
                #             - D2 * ( I - (D2-D1)*(D1+D2)^-1 ) * bz[n:] )

                blas.gemv(Asc, x, v)
                lapack.potrs(S, v)

                # x[:n] = D^-1 * ( rhs - A'*v ).
                blas.gemv(Asc, v, x, alpha=-1.0, beta=1.0, trans='T')
                x[:n] = div(x[:n], ds)

                # x[n:] = (D1+D2)^-1 * ( bx[n:] - D1*bz[:n]  - D2*bz[n:] )
                #         - (D2-D1)*(D1+D2)^-1 * x[:n]
                x[n:] = div(x[n:] - mul(d1, z[:n]) - mul(d2, z[n:]), d1 + d2) \
                        - mul(d3, x[:n])

                # z[:n] = D1^1/2 * (  x[:n] - x[n:] - bz[:n] )
                # z[n:] = D2^1/2 * ( -x[:n] - x[n:] - bz[n:] ).
                z[:n] = mul(W['di'][:n], x[:n] - x[n:] - z[:n])
                z[n:] = mul(W['di'][n:], -x[:n] - x[n:] - z[n:])

            return g

        solvers.options['show_progress'] = False

        c = solvers.coneqp(P, q, G, h, kktsolver=Fkkt)['x'][:n]
        c = np.squeeze(np.array(c))
        return c

    def gd(self, x, xcost, xgrad, stepsize=10, max_iter=100, verbosity=0):

        # gradient descent on x, with cost xcost and gradient xgrad

        if verbosity > 0:
            print("\ncost with initial u:", xcost(x))

        updated_once = False

        for ii in range(max_iter):

            if ii == 0:
                gx = xgrad(x)
                cx = xcost(x)

            # update
            x_update = x - stepsize * gx
            cx_new = xcost(x_update)

            if verbosity > 0:
                print("new cost on round", ii, xcost(x_update))

            if np.any(np.isnan(x_update)):
                if verbosity > 0:
                    print("diminishing stepsize (nan encountered)")
                stepsize = 0.1*stepsize
            # add a check: did the loss diminish?
            elif cx < cx_new:
                if verbosity > 0:
                    print("but not updated")
                stepsize = 0.1*stepsize  # if not reduce stepsize; probably overshot it
            else:
                if verbosity > 0:
                    print("updated")
                # update the previous
                diff = np.linalg.norm(x - x_update) / np.linalg.norm(x_update)

                x = np.copy(x_update)
                cx = cx_new

                # perhaps decrease the stepsize
                if ii >= 1 and updated_once:
                    if diff > prev_diff:
                        if verbosity > 1:
                            print("modifying step size!", 0.1 * stepsize)
                        stepsize = 0.1 * stepsize
                prev_diff = diff

                updated_once = True

                # check for convergence
                if ii > 0 and diff < 1e-4:
                    # print("stuck here")
                    break

                gx = xgrad(x)

            # also break if stepsize is really tiny
            if ii > 1 and stepsize < 1e-12:
                # print("no, stuck here")
                break

            if verbosity > 0:
                print("cost after round", ii, xcost(x))

        if verbosity > 0:
            print("final cost", xcost(x))
            print("final x:")
            print(x)

        return x

    def prox_gd(self, x, xcost, xgrad, lmbda, stepsize=10, max_iter=100, verbosity=0):

        def proximal1(x):
            ans = np.zeros(x.shape)
            inds1 = np.where(x >= lmbda)[0]
            inds2 = np.where(x <= -lmbda)[0]
            ans[inds1] = x[inds1] - lmbda
            ans[inds2] = x[inds2] + lmbda
            return ans

        # gradient descent on x, with cost xcost and gradient xgrad

        if verbosity > 0:
            print("\ncost with initial u:", xcost(x))

        updated_once = False

        for ii in range(max_iter):

            # update
            x_update = proximal1(x - stepsize * xgrad(x))  # basic gd for debugging
            diff = np.linalg.norm(x - x_update) / np.linalg.norm(x_update)

            if verbosity > 0:
                print("new cost on round", ii, xcost(x_update))

            # add a check: did the loss diminish?
            if xcost(x) < xcost(x_update):
                if verbosity > 0:
                    print("but not updated")
                stepsize = 0.1*stepsize  # if not reduce stepsize; probably overshot it
            else:
                if verbosity > 0:
                    print("updated")
                # update the previous
                x = np.copy(x_update)

                # perhaps decrease the stepsize
                if ii >= 1 and updated_once:
                    if diff > prev_diff:
                        if verbosity > 1:
                            print("modifying step size!", 0.1 * stepsize)
                        stepsize = 0.1 * stepsize
                prev_diff = diff

                updated_once = True

                # check for convergence
                if ii > 0 and diff < 1e-4:
                    # print("stuck here")
                    break

            # also break if stepsize is really tiny
            if ii > 1 and stepsize < 1e-12:
                # print("no, stuck here")
                break

            if verbosity > 0:
                print("cost after round", ii, xcost(x))

        if verbosity > 0:
            print("final cost", xcost(x))
            print("final x:")
            print(x)

        return x


# todo
#  c loss changes

class SPKM(baseSPKM):

    def __init__(self):
        super().__init__()

    def train(self, X, k, y, n_u, P, gamma, classification=True, loss=CosLoss, c=None, closs="sq", creg=2,
              init="randn", n_inits=5, rseed=0,
              standardise=False, calculate_new_rbf_param=False, standardise_labels=False,
              max_outer_iters=1, max_gd_iters=50, stepsize=10, verbosity=0):

        """
        For binary classification: labels with cosine loss should be -1 and 1
        For regression: data standardisation is recommended. In this case it is possible to re-calculate the RBF kernel
        parameter, since it is common to calculate it from mean of distances and data standardisation messes up with
        that.


        :param X:
        :param k:
        :param y:
        :param n_u:
        :param P1:
        :param gamma:
        :param lmbda:
        :param loss:
        :param c:
        :param closs:
        :param init:
        :param n_inits:
        :param standardise:
        :param rseed:
        :param max_outer_iters:
        :param max_gd_iters:
        :param stepsize:
        :param verbosity:
        :return:
        """

        Y = np.copy(y)

        self.kernel = k

        self.standardise = standardise
        if self.standardise:
            self.data_means = np.mean(X, axis=0)
            self.data_stds = np.std(X, axis=0)
            X = (X-self.data_means)/self.data_stds

            if calculate_new_rbf_param:  # I don't want to do this always - n^2 costly
                from kernels_and_gradients_old import RBFKernel
                if isinstance(k, RBFKernel):
                    from sklearn.metrics import pairwise_distances
                    # print(np.mean(pairwise_distances(X)))
                    self.kernel = RBFKernel(np.mean(pairwise_distances(X)))

        self.standardise_labels = False
        if not classification:
            if standardise_labels:
                self.standardise_labels = True
                self.y_mean = np.mean(Y)
                self.y_std = np.std(Y)
                Y = (Y - self.y_mean) / self.y_std

        if c is not None:
            c = c[:n_u]  # in case c given is longer than needed
        else:
            if classification:
                # this piece of code gives array where each pair has half the significance to the next one.
                divarray = np.arange(1, np.ceil(n_u/2)+1, 1)
                tmp_ones = np.ones(int(np.ceil(n_u/2)))
                tmp_ones = tmp_ones / (2 ** (divarray - 1))
                # tmp_ones = tmp_ones/divarray
                c = []
                for ii in tmp_ones:
                    c.append(ii)
                    c.append(-ii)
                c = np.array(c)[:n_u]

            initial_c = np.copy(c)

        [n, d1] = X.shape

        yms = [0.001, 1, 0.00001]#, 0.1, 10]

        if init == "const":
            n_inits = 1

        # ==========================================================================================================0
        # repeat the algorithm "n_inits" times with different initialisations
        for init_loop in range(n_inits):

            if verbosity > 0:
                print("\nStarting loop", init_loop, "\n")

            # random initialization but with reproducible results
            np.random.seed(rseed + init_loop)
            if not classification:
                np.random.seed(rseed + init_loop// len(yms))

            if classification:
                c = np.copy(initial_c)
            else:
                yperm = np.random.permutation(len(y))
                c = y[yperm][:n_u]
                c *= yms[init_loop % len(yms)]

            if init == "const":
                u = np.zeros((n_u, d1))
                if classification:
                    c += 0.1  # if I use constant init and have a and -a in c it causes nan values
            if init == "randn" or ((init == "mix") and (init_loop % 2 == 1)):
                u = np.random.randn(n_u, d1)
            if init == "rand":
                u = np.random.rand(n_u, d1)
            if init == "data" or ((init == "mix") and (init_loop % 2 == 0)):
                if classification:
                    u = np.zeros((n_u, d1))
                    pinds = np.where(np.array(c) > 0)[0]
                    ninds = np.where(np.array(c) < 0)[0]
                    ypinds = np.where(Y > 0)[0]
                    yninds = np.where(Y < 0)[0]
                    u[pinds, :] = np.squeeze(X[ypinds, :][np.random.permutation(len(ypinds))[:len(pinds)], :])
                    u[ninds, :] = np.squeeze(X[yninds, :][np.random.permutation(len(yninds))[:len(ninds)], :])
                else:
                    u = X[yperm[:n_u], :]

            # if mask and (init=="randn" or init=="rand"):
            #     # obtain the mask for u:
            #     u_mask = np.where((np.sum(X, axis=0))==0)[0]
            #     for ii in range(n_u):
            #         u[ii, u_mask] = 0

            # ---------------------------------------- loss function -------------------------------------------

            # dloss in suitable format for gd
            myloss = loss(X, self.kernel, Y)

            def grad_u(u):
                return myloss.grad_u(u, c)

            def cost_u(u):
                val = myloss.cost_uc(u, c)
                return val

            def cost_uc(u, c):
                val = myloss.cost_uc(u, c)
                return val

            # ------------------------------------- iterate over u and c ----------------------------------------

            # u = np.squeeze(u)
            if P > 0:
                for ii in range(n_u):
                    u[ii, :] = proj_l1(u[ii, :], P)

            for outer_loop in range(max_outer_iters):

                # gradient based approach

                prev_u = np.copy(u)
                prev_c = np.copy(c)

                # ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ u update ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤

                if gamma == 0:
                    u = self.gd(u, cost_u, grad_u, stepsize=stepsize, max_iter=max_gd_iters, verbosity=verbosity)
                else:
                    # print("I am doing the proximal gradient updates")
                    u = self.prox_gd(u, cost_u, grad_u, gamma)

                # projection to l1 ball
                if P > 0:
                    for ii in range(u.shape[0]):
                        u[ii, :] = proj_l1(u[ii, :], P)

                if self.kernel.zeroone:
                    ku = self.kernel.kernel(X, u) * 2 - 1
                else:
                    ku = self.kernel.kernel(X, u)

                # ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ c update ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤

                if closs == "sq":
                    if creg == 1:
                        c = self.solve_cone_c(ku, y)
                        c = np.squeeze(np.array(c))
                        try:
                            len(c)
                        except TypeError:
                            c = [c]
                    elif creg == 2:
                        lmbda = 0.001
                        if n_u > 1:
                            c = 2 * np.dot(np.linalg.pinv(2 * np.dot(ku.T, ku) + lmbda*np.eye(n_u)), np.dot(Y, ku))
                        else:
                            ku = ku[:, np.newaxis]
                            c = 2 * (1/(2 * np.dot(ku.T, ku) + lmbda)*np.dot(Y, ku))
                elif closs == "cos":
                    mycloss = CosLoss(X, self.kernel, Y)

                    if creg == 2:
                        def grad_c(c):
                            return mycloss.grad_c(c, u) + 0.001*c

                        def cost_c(c):
                            val = mycloss.cost_uc(u, c) + 0.001*np.linalg.norm(c)**2
                            return val
                        c = self.gd(c, cost_c, grad_c, stepsize=stepsize, max_iter=max_gd_iters, verbosity=verbosity)
                    else:
                        def grad_c(c):
                            return mycloss.grad_c(c, u)

                        def cost_c(c):
                            val = mycloss.cost_uc(u, c)
                            return val
                        c = self.prox_gd(c, cost_c, grad_c, gamma)

                # ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ stopping criteria if looping over u and c steps ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
                # if number of the outer loops is high, need to have some stopping criteria
                if outer_loop > 2:
                    udiff = np.linalg.norm(u-prev_u)/np.linalg.norm(prev_u)
                    cdiff = np.linalg.norm(c-prev_c)/np.linalg.norm(prev_c)
                    if (udiff < 1e-3) and (cdiff < 1e-3):
                        break

            # get the best result of the inits
            if init_loop == 0:
                self.u = np.copy(u)
                self.c = np.copy(c)
            else:
                if cost_uc(self.u, self.c) > cost_uc(u, c):
                    self.u = np.copy(u)
                    self.c = np.copy(c)

        if standardise:
            self.u_unstandardised = self.u*self.data_stds+self.data_means

        return self.u, self.c

    def _unstandardise_labels(self, p):
        if self.standardise_labels:
            return p * self.y_std + self.y_mean
        else:
            return p

    def predict(self, Xt):
        kvec = self.kernel.kernel(Xt, np.squeeze(self.u))
        if self.kernel.zeroone:
            kvec = kvec*2 - 1
        c = np.atleast_1d(np.array(self.c))
        if len(c) == 1:
            return self._unstandardise_labels(c[0] * kvec)
        return self._unstandardise_labels(np.dot(self.c, kvec.T))


class MCSPKM(baseSPKM):

    def __init__(self):
        super().__init__()

    def train(self, X, k, y, n_u, P, gamma, classification=True, loss=MCCosLoss, c=None, closs="sq", creg=2,
              init="randn", n_inits=5, rseed=0,
              standardise=False, calculate_new_rbf_param=False, standardise_labels=False,
              max_outer_iters=1, max_gd_iters=50, stepsize=10, verbosity=0):

        """
        Multi-class version of SPKM
        
        :param X: 
        :param k: 
        :param y: 
        :param n_u: 
        :param P: 
        :param gamma: 
        :param classification: 
        :param loss: 
        :param c: 
        :param closs: 
        :param init: 
        :param n_inits: 
        :param standardise: 
        :param rseed: 
        :param max_outer_iters: 
        :param max_gd_iters: 
        :param stepsize: 
        :param verbosity: 
        :return: 
        """

        Y = np.copy(y)
        
        unique_labels = np.sort(np.unique(Y))
        Yonehot = -np.ones((len(Y), len(unique_labels)))
        for ul in range(len(unique_labels)):
            inds = np.where(Y == unique_labels[ul])[0]
            Yonehot[inds, ul] = 1

        self.unique_labels = unique_labels

        nc = len(unique_labels)  # number of classes

        if c is not None:
            c = c[:(n_u*nc), :]
        else:

            c = -np.ones((n_u*nc, nc))

            # sample to be similar to one class
            for nnc in range(nc):
                c[nnc:(nnc+1), nnc] = 1*nc  # first ten just that one class

            # then require the sample to be similar to two classes
            if n_u > 1:
                indx = len(unique_labels)
                for ii in range(len(unique_labels)):
                    for jj in range(ii):
                        if indx < nc*n_u:
                            c[indx, ii] = nc/2
                            c[indx, jj] = nc/2
                            indx +=1
                # could do the same with three samples, for now not implemented

        self.kernel = k

        [n, d1] = X.shape

        if init == "const":
            n_inits = 1
            
        for loop in range(n_inits):  # todo change this back
            if verbosity > 0:
                print("\nStarting loop", loop, "\n")

            # random initialization but with reproducible results
            np.random.seed(rseed + loop)

            if init == "const":
                u = np.zeros((n_u*nc, d1))
            if init == "randn" or ((init == "mix") and (loop % 2 == 1)):
                u = np.random.randn(n_u*nc, d1)
            if init == "rand":
                u = np.random.rand(n_u*nc, d1)
            if init == "data" or ((init == "mix") and (loop % 2 == 0)):
                u = np.zeros((n_u*nc, d1))
                for nnc in range(nc):
                    pinds = np.where(np.array(c[:, nnc]) > 0)[0]
                    ppinds = np.where(Yonehot[:, nnc] > 0)[0]
                    u[pinds, :] = np.squeeze(X[ppinds, :][np.random.permutation(len(ppinds))[:len(pinds)], :])

            # if mask and (init=="randn" or init=="rand"):
            #     # obtain the mask for u:
            #     u_mask = np.where((np.sum(X, axis=0))==0)[0]  # assume this would work
            #     for ii in range(n_u*nc):
            #         u[ii, u_mask] = 0

            u = np.squeeze(u)

            if P > 0:
                if init != "opt" and loop != 0:
                    for ii in range(n_u):
                        u[ii, :] = proj_l1(u[ii, :], P)

            # ------------- define the loss -------------

            if loss == "cos":
                myloss = MCCosLoss(X, self.kernel, Yonehot)

            # define in suitable format for gd

            def grad_u(u):
                return myloss.grad_u(u, c)

            def cost_u(u):
                val = myloss.cost_uc(u, c)
                print("cost:", val)
                return val

            def cost_uc(u, c):
                return myloss.cost_uc(u, c)

            # ------------- loop u and c steps -------------
            for outer_loop in range(max_outer_iters):

                prev_u = np.copy(u)
                prev_c = np.copy(c)

                # ¤¤¤¤¤¤¤¤ u update ¤¤¤¤¤¤¤¤

                if gamma == 0:
                    u = self.gd(u, cost_u, grad_u, stepsize=stepsize, max_iter=max_gd_iters, verbosity=verbosity)
                else:
                    u = self.prox_gd(u, cost_u, grad_u, gamma)

                if P > 0:
                    for ii in range(u.shape[0]):
                        u[ii, :] = proj_l1(u[ii, :], P)


                # ¤¤¤¤¤¤¤¤ c update ¤¤¤¤¤¤¤¤

                u = np.squeeze(u)
                if self.kernel.zeroone:
                    ku = self.kernel.kernel(X, u) * 2 - 1
                else:
                    ku = self.kernel.kernel(X, u)

                c = np.zeros(c.shape)
                # solve for each class individually
                for nnc in range(nc):
                    if closs == "sq":
                        if creg == 1:
                            ctmp = self.solve_cone_c(ku, Yonehot[:, nnc])
                            ctmp = np.squeeze(np.array(ctmp))
                        elif creg == 2:
                            ctmp = 2 * np.dot(np.linalg.pinv(2 * np.dot(ku.T, ku) +
                                                             0.001*np.eye(ku.shape[1])),
                                              np.dot(Yonehot[:, nnc], ku))
                    c[:, nnc] = np.copy(ctmp)

                # ¤¤¤¤¤¤¤¤ stopping criteria if u and c are iterated over ¤¤¤¤¤¤¤¤

                if outer_loop > 2:
                    udiff = np.linalg.norm(u - prev_u) / np.linalg.norm(prev_u)
                    cdiff = np.linalg.norm(c - prev_c) / np.linalg.norm(prev_c)
                    if udiff < 1e-3 and cdiff < 1e-3:
                        break

            if loop == 0:
                self.u = np.copy(u)
                self.c = np.copy(c)
            elif cost_uc(u, c) < cost_uc(self.u, self.c):
                self.u = np.copy(u)
                self.c = np.copy(c)

        return self.u

    def predict(self, Xt):
        kvec = self.kernel.kernel(Xt, np.squeeze(self.u))
        if self.kernel.zeroone:
            onehotpreds = np.dot(kvec * 2 - 1, self.c)
        else:
            onehotpreds = np.dot(kvec, self.c)
        # take the largest value
        return self.unique_labels[np.argmax(onehotpreds, axis=1)]


class MVSPKM(baseSPKM):

    def __init__(self):
        super().__init__()

    def train(self, X, k, y, n_u, P, gamma, classification=True, loss=MVCosLoss, c=None, closs="sq", creg=2,
              init="randn", n_inits=5, rseed=0,
              standardise=False, calculate_new_rbf_param=False, standardise_labels=False,
              max_outer_iters=1, max_gd_iters=50, stepsize=10, verbosity=0):

        """
        Multi-view extension of SPKM: the X comes potentially from multiple views (a dictionary) that are applied to
        different kernels in k. Labels Y are shared accross views.
        Note: dictionary keys are assumed to be integers 0-(V-1)!
        Number of u's (basis vectors) searched for can also potentially vary between views.
        Note: implementation is not very efficient if for example X is same but applied with multiple kernels.

        :param X:
        :param k:
        :param y:
        :param n_u:
        :param P:
        :param gamma:
        :param classification:
        :param loss:
        :param c:
        :param closs:
        :param creg:
        :param init:
        :param n_inits:
        :param standardise:
        :param rseed:
        :param max_outer_iters:
        :param max_gd_iters:
        :param stepsize:
        :param verbosity:
        :return:
        """

        n_views = len(X.keys())  # assume X is dictionary; since the views can be of different dimensionalities

        if isinstance(n_u, list):
            assert n_views == len(n_u)
            n_us = np.array(n_u)
        else:
            n_us = np.ones(n_views) * n_u

        if isinstance(P, list):
            assert n_views == len(P)
        else:
            P = np.ones(n_views) * P

        if isinstance(k, list):
            assert n_views == len(k)
        else:
            k_tmp = np.empty(n_views)
            for vv in range(n_views):
                k_tmp[vv] = k
            k = k_tmp

        n_us = n_us.astype(int)

        self.n_us = n_us

        nn_us = np.sum(n_us)
        n = len(y)

        Y = np.copy(y)

        if c is not None:
            tmp_c = {}
            for vv in range(n_views):
                tmp_c[vv] = c[vv][:n_us[vv]]
            c = tmp_c
        else:

            c = {}

            # this piece of code gives array where each pair has half the significance to the next one.
            divarray = np.arange(1, 11, 1)
            tmp_vec = np.ones(10)
            tmp_vec = tmp_vec / (2 ** (divarray - 1))
            # tmp_ones = tmp_ones/divarray
            tmp_vec2 = []
            for ii in tmp_vec:
                tmp_vec2.append(ii)
                tmp_vec2.append(-ii)

            for vv in range(n_views):
                c[vv] = np.array(tmp_vec2)[:n_us[vv]]

        c_vec = []
        for vv in range(n_views):
            c_vec.extend(c[vv])
        c_vec = np.array(c_vec)
        self.c_vec = c_vec

        self.kernels = k

        ures = {}

        if init == "const":
            n_inits = 1

        for loop in range(n_inits):

            # todo regression inits

            # ----------------------- initialisation -----------------------

            # random initialization but with reproducible results
            np.random.seed(rseed + loop)
            u = {}
            if init == "randn":
                for vv in range(n_views):
                    u[vv] = np.random.randn(n_us[vv], X[vv].shape[1])
            elif init == "data":
                for vv in range(n_views):
                    # u[vv] = np.squeeze(X[vv][np.random.permutation(X[vv].shape[0])[:n_us[vv]], :])
                    pinds = np.where(np.array(c[vv]) > 0)[0]
                    ninds = np.where(np.array(c[vv]) < 0)[0]
                    u[vv][pinds, :] = np.squeeze(
                        X[vv][np.where(Y > 0)[0], :][np.random.permutation(len(np.where(Y > 0)[0]))[:len(pinds)],
                        :])
                    u[vv][ninds, :] = np.squeeze(
                        X[vv][np.where(Y < 0)[0], :][np.random.permutation(len(np.where(Y < 0)[0]))[:len(ninds)],
                        :])
            else:
                for vv in range(n_views):
                    u[vv] = np.random.randn(n_us[vv], X[vv].shape[1])

            for vv in range(n_views):
                if P[vv] != 0:
                    for ii in range(n_us[vv]):
                        u[vv][ii, :] = proj_l1(u[vv][ii, :], P[vv])

            # ----------------------- loss function -----------------------

            if loss == "cos":
                from losses_and_gradients import MVCosLoss
                myloss = MVCosLoss(X, self.kernels, Y)

            def grad_u(u):
                return myloss.grad_u(u, c)

            def cost_u(u):
                val = myloss.cost_uc(u, c)
                print("cost:", val)
                return val

            def cost_uc(u, c):
                return myloss.cost_uc(u, c)

            # ----------------------- solve for u and c -----------------------

            for outer_loop in range(max_outer_iters):

                prev_u = u  # todo np copy & dictionaries? references?
                prev_c_vec = np.copy(c_vec)

                # ¤¤¤¤¤¤¤¤¤¤ u solution ¤¤¤¤¤¤¤¤¤¤

                if gamma == 0:
                    u = self.gd_dict(u, cost_u, grad_u,
                                     max_iter=max_gd_iters)  # note u is a dictionary so need to use this one
                else:
                    u = self.prox_gd_dict(u, cost_u, grad_u, gamma)

                for vv in range(n_views):
                    if P[vv] != 0:
                        for ii in range(n_us[vv]):
                            u[vv][ii, :] = proj_l1(u[vv][ii, :], P[vv])

                # ¤¤¤¤¤¤¤¤¤¤ c solution ¤¤¤¤¤¤¤¤¤¤

                kuall = np.zeros((n, nn_us))
                indx = 0
                for vv in range(n_views):
                    utmp = np.squeeze(u[vv])
                    if self.kernels[vv].zeroone:
                        ku = self.kernels[vv].kernel(X[vv], utmp) * 2 - 1
                    else:
                        ku = self.kernels[vv].kernel(X[vv], utmp)
                    kuall[:, indx:(indx + n_us[vv])] = ku
                    indx += n_us[vv]

                if closs == "sq":
                    if creg == 1:
                        c_vec = self.solve_cone_c(kuall, Y)
                    elif creg == 2:
                        c_vec = 2 * np.dot(np.linalg.pinv(2 * np.dot(kuall.T, kuall) +
                                                          0.001*np.eye(kuall.shape[1])), np.dot(Y, kuall))
                self.c_vec = c_vec

                # ¤¤¤¤¤¤¤¤¤¤ if iterated, check convergence ¤¤¤¤¤¤¤¤¤¤

                if outer_loop > 2:
                    udiff = np.linalg.norm(u - prev_u) / np.linalg.norm(prev_u)  # todo !!!!!!!!!!!!!111
                    cdiff = np.linalg.norm(c_vec - prev_c_vec) / np.linalg.norm(prev_c_vec)
                    if udiff < 1e-3 and cdiff < 1e-3:
                        break

            if loop == 0:
                self.u = np.copy(u)
                self.c_vec = np.copy(c_vec)
            elif cost_uc(u, c_vec) < cost_uc(self.u, self.c_vec):
                self.u = np.copy(u)
                self.c_vec = np.copy(c_vec)

        return self.u

    def gd_dict(self, x, xcost, xgrad, stepsize=10, max_iter=100, verbosity=0):

        # Need to add to the basic gd as here I use dictionaries
        n_views = len(x.keys())

        updated_once = False

        for ii in range(max_iter):

            if ii == 0:
                gx = xgrad(x)
                cx = xcost(x)

            # update
            u_update = {}
            diff = 0
            # print(ii, "\nu",u)
            for vv in range(n_views):
                u_update[vv] = x[vv] - stepsize * gx[vv]  # basic gd for debugging
                diff += np.linalg.norm(x[vv] - u_update[vv]) / np.linalg.norm(u_update[vv])

            cx_new = xcost(u_update)

            # add a check: did the loss diminish?
            if cx < cx_new:
                stepsize = 0.1 * stepsize  # if not reduce stepsize; probably overshot it
            else:

                for vv in range(n_views):
                    diff += np.linalg.norm(x[vv] - u_update[vv]) / np.linalg.norm(u_update[vv])

                # update the previous
                x = u_update.copy()
                cx = xcost(x)

                # perhaps decrease the stepsize
                if ii >= 1 and updated_once:
                    if diff > prev_diff:
                        if verbosity > 1:
                            print("modifying step size!", 0.1 * stepsize)
                        stepsize = 0.1 * stepsize
                prev_diff = diff
                updated_once = True

                # check for convergence
                if ii > 0 and diff < 1e-6:
                    # print("stuck here")
                    break

                gx = xgrad(x)

            # also break if stepsize is really tiny
            if ii > 1 and stepsize < 1e-12:
                # print("no, stuck here")
                break

            if verbosity > 0:
                print(ii, xcost(x))

        return x

    def prox_gd_dict(self, x, xcost, xgrad, lmbdas, stepsize=10, max_iter=100, verbosity=0):

        # Need to add to the basic gd as here I use dictionaries
        n_views = len(x.keys())

        try:
            len(lmbdas)
        except:  # if it's just one value
            lmbdas = np.ones(n_views) * lmbdas

        def proximal1(x, lmbda):
            ans = np.zeros(x.shape)
            inds1 = np.where(x >= lmbda)
            inds2 = np.where(x <= -lmbda)
            ans[inds1] = x[inds1] - lmbda
            ans[inds2] = x[inds2] + lmbda
            return ans

        updated_once = False

        for ii in range(max_iter):

            # update
            u_update = {}
            diff = 0
            # print(ii, "\nu",u)
            tmp = xgrad(x)
            for vv in range(n_views):
                u_update[vv] = proximal1(x[vv] - stepsize * tmp[vv], lmbdas[vv])
                diff += np.linalg.norm(x[vv] - u_update[vv]) / np.linalg.norm(u_update[vv])

            # add a check: did the loss diminish?
            if xcost(x) < xcost(u_update):
                stepsize = 0.1 * stepsize  # if not reduce stepsize; probably overshot it
            else:
                # update the previous
                x = u_update.copy()

                # perhaps decrease the stepsize
                if ii >= 1 and updated_once:
                    if diff > prev_diff:
                        if verbosity > 1:
                            print("modifying step size!", 0.1 * stepsize)
                        stepsize = 0.1 * stepsize
                prev_diff = diff
                updated_once = True

                # check for convergence
                if ii > 0 and diff < 1e-6:
                    # print("stuck here")
                    break

            # also break if stepsize is really tiny
            if ii > 1 and stepsize < 1e-12:
                # print("no, stuck here")
                break

            if verbosity > 0:
                print(ii, xcost(x))

        return x

    def predict(self, Xt):

        kvecall = np.zeros((Xt[0].shape[0], np.sum(self.n_us)))
        indx = 0
        for vv in range(len(self.n_us)):
            utmp = np.squeeze(self.u[vv])
            if self.kernels[vv].zeroone:
                ku = self.kernels[vv].kernel(Xt[vv], utmp) * 2 - 1
            else:
                ku = self.kernels[vv].kernel(Xt[vv], utmp)
            kvecall[:, indx:(indx + self.n_us[vv])] = ku
            indx += self.n_us[vv]

        return np.dot(kvecall, self.c_vec)


class SPKMovoRBF(baseSPKM):

    def __init__(self):
        super().__init__()

    def train(self, X, k, y, n_u, P, gamma, classification=True, loss=CosLoss, c=None, closs="sq", creg=2,
              init="randn", n_inits=5, rseed=0,
              standardise=False, calculate_new_rbf_param=False, standardise_labels=False,
              max_outer_iters=1, max_gd_iters=50, stepsize=10, verbosity=0):

        from sklearn.metrics import pairwise_distances
        from kernels_and_gradients_old import RBFKernel

        # OVO strategy
        all_labels = np.sort(np.unique(y))
        self.all_labels = all_labels

        estimators = np.empty(shape=(len(all_labels), len(all_labels)), dtype=object)
        np.fill_diagonal(estimators, None)

        for ind, elem in enumerate(all_labels):
            inds1 = np.where(y==elem)[0]
            nelems1 = np.minimum(200, len(inds1))
            for ind2, elem2 in enumerate(all_labels):
                inds2 = np.where(y==elem2)[0]
                nelems2 = np.minimum(200, len(inds2))
                if ind < ind2:

                    inds12 = np.append(inds1, inds2)
                    inds12small = np.append(inds1[:nelems1], inds2[:nelems2])

                    # determine the suitable estimator parameter
                    rbf_param = np.mean(pairwise_distances(X[inds12small, :]))
                    estimator = SPKM()
                    # fit the estimator
                    tmpy = np.ones(len(inds12))
                    tmpy[y[inds12]==elem] = -1
                    estimator.train(X[inds12, :], RBFKernel(rbf_param), tmpy, n_u, P, gamma,
                                    classification=classification, loss=loss, c=c, closs=closs, creg=creg,
                                    init=init, n_inits=n_inits, standardise=standardise, rseed=rseed,
                                    max_outer_iters=max_outer_iters, max_gd_iters=max_gd_iters,
                                    stepsize=stepsize, verbosity=verbosity)
                    estimators[ind, ind2] = estimator

        self.estimators = estimators

    def predict(self, Xt):

        from scipy.stats import mode

        l = len(self.all_labels)

        preds = np.zeros((Xt.shape[0], int((l*l-l)/2)))

        indx = 0
        for ind, elem in enumerate(self.all_labels):
            for ind2, elem2 in enumerate(self.all_labels):
                if ind < ind2:
                    opreds = self.estimators[ind, ind2].predict(Xt)
                    preds[np.where(opreds<0)[0], indx] = elem
                    preds[np.where(opreds>0)[0], indx] = elem2
                    indx += 1

        return mode(preds, axis=1)[0].flatten()


class PairwiseSPKM(SPKM):

    def __init__(self):
        super().__init__()
        self.v = None

    def train(self, X, k, y, n_uv, P, gamma,
              classification=True, loss=PairwiseCosLoss, c=None, closs="sq", creg=2,
              init="randn", n_inits=5, rseed=0,
              standardise=False, calculate_new_rbf_param=False, standardise_labels=False,
              max_outer_iters=1, max_gd_iters=50, stepsize=10, verbosity=0):

        """
            Pairwise data

            :param X: Data in particular format for pairwise two-view data: X[0] and X[2] should contain the unique
            data samples (sizes n1*d1 and n2*d2), while X[1] and X[3] should contain indicies by which the pairs are
            formed from them; lenght n.
            :param y: length n vector of the labels
            :param k: k[0] and k[1] should contain the kernels
            :param max_outer_iters: vector of length 2 specifying max iterations for u&v, and (u&v)&c. if both are
            the same can be also given as one integer

            :return: u, v and c
            """

        # X, xinds, Z, zinds, kx, kz, y, n_u, P, gamma,

        V1 = X[0]
        ix1 = X[1]
        V2 = X[2]
        ix2 = X[3]
        k1 = k[0]
        k2 = k[1]

        if isinstance(max_outer_iters, list):
            max_uv_iters = max_outer_iters[0]
            max_uvc_iters = max_outer_iters[1]
        else:
            max_uv_iters = max_outer_iters
            max_uvc_iters = max_outer_iters

        if not isinstance(P, list):
            P = [P, P]

        self.kernel1 = k[0]
        self.kernel2 = k[1]

        self.standardise = standardise
        if self.standardise:
            self.data_V1_means = np.mean(V1, axis=0)
            self.data_V1_stds = np.std(V1, axis=0)
            V1 = (V1 - self.data_V1_means) / self.data_V1_stds
            self.data_V2_means = np.mean(V2, axis=0)
            self.data_V2_stds = np.std(V2, axis=0)
            # print(np.count_nonzero(V2))
            # print(self.data_V2_stds)
            self.data_V2_std_inds = np.where(self.data_V2_stds != 0)[0]
            V2 = (V2 - self.data_V2_means)
            V2[:, self.data_V2_std_inds] /= self.data_V2_stds[self.data_V2_std_inds]

            if calculate_new_rbf_param:  # I don't want to do this always - n^2 costly
                from kernels_and_gradients_old import RBFKernel
                from sklearn.metrics import pairwise_distances
                if isinstance(k1, RBFKernel):
                    # print(np.mean(pairwise_distances(X)))
                    param = np.mean(pairwise_distances(V1))
                    self.kernel1 = RBFKernel(param)
                if isinstance(k2, RBFKernel):
                    # print(np.mean(pairwise_distances(X)))
                    param = np.mean(pairwise_distances(V2))
                    self.kernel2 = RBFKernel(param)

        # ----------------------------------------------
        [n1, d1] = V1.shape
        [n2, d2] = V2.shape

        n = len(y)

        Y = np.copy(y)
        # ----------------------------------------------

        batch_size = 100

        # Also I checked that possibly random.permutation will be faster than choice with replace=False. Used:
        # timeit.timeit( 'permutation(10000)[:2000]', setup='from numpy.random import permutation', number=10000)
        # timeit.timeit( 'choice(10000, 2000, replace=False)', setup='from numpy.random import choice', number=10000)
        # and permutation was about 1.34 usually, choice was about 1.46

        # ----------------------------------------------

        if c is not None:
            initial_c = c[:n_uv]
        else:
            # this piece of code gives array where each pair has half the significance to the next one.
            divarray = np.arange(1, 11, 1)
            tmp_ones = np.ones(10)
            tmp_ones = tmp_ones/(2**(divarray-1))
            # tmp_ones = tmp_ones/divarray
            tmp_ones2 = []
            for ii in tmp_ones:
                tmp_ones2.append(ii)
                tmp_ones2.append(-ii)

            initial_c = np.array(tmp_ones2)[:n_uv]

        # # here a re-weighting scheme!! if there is label imbalance
        # if balance_classes:
        #     n1 = len(np.where(Y == -1)[0])
        #     n2 = len(np.where(Y == 1)[0])
        #     if n2 < n1:
        #         Y[np.where(Y == 1)[0]] = n1 / n2
        #     if n1 < n2:
        #         Y[np.where(Y == -1)[0]] = n2 / n1

        yms = [0.001, 1, 0.00001]#, 0.1, 10]

        for init_loop in range(n_inits):

            # random initialization but with reproducible results
            np.random.seed(rseed + init_loop)
            if not classification:
                np.random.seed(rseed + init_loop // len(yms))

            if classification:
                c = np.copy(initial_c)
            else:
                yperm = np.random.permutation(len(y))
                c = y[yperm][:n_uv]
                c *= yms[init_loop % len(yms)]

            if init == "const":
                u = np.zeros((n_uv, d1))
                v = np.zeros((n_uv, d2))
                if classification:
                    c += 0.1  # if I use constant init and have a and -a in c it causes nan values
            if init == "randn" or ((init == "mix") and (init_loop % 2 == 1)):
                u = np.random.randn(n_uv, d1)
                v = np.random.randn(n_uv, d2)
            if init == "rand":
                u = np.random.rand(n_uv, d1)
                v = np.random.rand(n_uv, d2)
            if init == "data" or ((init == "mix") and (init_loop % 2 == 0)):
                if classification:
                    u = np.zeros((n_uv, d1))
                    pinds = np.where(np.array(c) > 0)[0]
                    ninds = np.where(np.array(c) < 0)[0]
                    ypinds = np.where(Y > 0)[0]
                    yninds = np.where(Y < 0)[0]
                    u[pinds, :] = np.squeeze(V1[ypinds, :][np.random.permutation(len(ypinds))[:len(pinds)], :])
                    u[ninds, :] = np.squeeze(V1[yninds, :][np.random.permutation(len(yninds))[:len(ninds)], :])
                    v[pinds, :] = np.squeeze(V2[ypinds, :][np.random.permutation(len(ypinds))[:len(pinds)], :])
                    v[ninds, :] = np.squeeze(V2[yninds, :][np.random.permutation(len(yninds))[:len(ninds)], :])
                else:
                    u = V1[yperm[:n_uv], :]
                    v = V2[yperm[:n_uv], :]

            # u = np.squeeze(u)

            # --------------------------------------
            if P[0] > 0:
                if init != "opt" and init_loop != 0:
                    for ii in range(n_uv):
                        u[ii, :] = proj_l1(u[ii, :], P[0])
            if P[1] > 0:
                if init != "opt" and init_loop != 0:
                    for ii in range(n_uv):
                        v[ii, :] = proj_l1(v[ii, :], P[1])

            # --------------------------------------
            # get the random order for the batches
            order = np.random.permutation(n)
            # fill out the last batch randomly
            leftover = n % batch_size
            n_batches = n//batch_size
            if leftover != 0:
                n_batches += 1
                order = np.append(order, order[np.random.permutation(n)][:(batch_size-leftover)])

            # --------------------------------------
            myloss = loss(X, k, Y)

            def cost_u(u):
                return myloss.cost_uvc(u, v, c)
            def cost_v(v):
                return myloss.cost_uvc(u, v, c)
            def cost_uv(u, v):
                return myloss.cost_uvc(u, v, c)
            def cost_uvc(u, v, c):
                return myloss.cost_uvc(u, v, c)
            def cost_c(c):
                return myloss.cost_uvc(u, v, c)
            def grad_u(u):
                return myloss.grad_u(u, v, c)
            def grad_v(v):
                return myloss.grad_v(v, u, c)

            def batch_cost_u(u, elems):
                return myloss.cost_uvc(u, v, c, elems=elems)
            def batch_cost_v(v, elems):
                return myloss.cost_uvc(u, v, c, elems=elems)
            def batch_cost_uv(u, v, elems):
                return myloss.cost_uvc(u, v, c, elems=elems)
            def batch_cost_uvc(u, v, c, elems):
                return myloss.cost_uvc(u, v, c, elems=elems)
            def batch_cost_c(c, elems):
                return myloss.cost_uvc(u, v, c, elems=elems)
            def batch_grad_u(u, elems):
                return myloss.grad_u(u, v, c, elems=elems)
            def batch_grad_v(v, elems):
                return myloss.grad_v(v, u, c, elems=elems)

            # --------------------------------------
            for uvc_iter in range(max_uvc_iters):

                # ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ update u and v before c ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤

                # print("n, batch size, n batches:")
                # print(n, batch_size, n_batches)
                # import time
                # from datetime import timedelta

                for uv_iter in range(max_uv_iters):

                    # print("u before u update:", u)
                    # print("v before u update:", v)

                    # ----------------- u -----------------
                    # print("\n\n - u update -\n")
                    # u_update = self.gd(u, cost_u, grad_u, max_iter=10)
                    # t0 = time.clock()
                    u_update = self.batch_gd(u, batch_cost_u, batch_grad_u, order,
                                             batch_size=batch_size, max_iter=10)
                    # print("u update took ", timedelta(seconds=(time.clock()-t0)))
                    udiff = np.linalg.norm(u - u_update) / np.linalg.norm(u_update)
                    u = u_update

                    if verbosity > 0:
                        print("cost after u update in round", ii, cost_uv(u, v))

                    # print("u after u update:", u)
                    # print("v after u update:", v)

                    if P[0] > 0:
                        for ii in range(n_uv):
                            u[ii, :] = proj_l1(u[ii, :], P[0])

                    # ----------------- v -----------------
                    # print("\n\n - v update -\n")
                    # v_update = self.gd(v, cost_v, grad_v, max_iter=10)
                    # t0 = time.clock()
                    v_update = self.batch_gd(v, batch_cost_v, batch_grad_v, order,
                                             batch_size=batch_size, max_iter=10)
                    # print("v update took ", timedelta(seconds=(time.clock()-t0)))
                    vdiff = np.linalg.norm(v - v_update) / np.linalg.norm(v_update)
                    v = v_update

                    # print("u after v update:", u)
                    # print("v after v update:", v)
                    # projection to l1 or l2 ball? or directly optimizing in a manifold
                    if P[1] > 0:
                        for ii in range(n_uv):
                            v[ii, :] = proj_l1(v[ii, :], P[1])

                    if verbosity > 0:
                        print("cost after v update in round", ii, cost_uv(u, v), udiff, vdiff)

                    # 3) converged?
                    if udiff < 1e-4 and vdiff < 1e-4:
                        break

                # ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ c update ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤

                # t0 = time.clock()
                u = np.squeeze(u)
                if k1.zeroone:
                    ku = k[0].kernel(V1[ix1, :], u) * 2 - 1
                else:
                    ku = k[0].kernel(V1[ix1, :], u)
                v = np.squeeze(v)
                if k2.zeroone:
                    kv = k[1].kernel(V2[ix2, :], v) * 2 - 1
                else:
                    kv = k[1].kernel(V2[ix2, :], v)
                kuv = ku * kv

                if closs == "sq":
                    if creg == 1:
                        c = self.solve_cone_c(kuv, y)
                        c = np.squeeze(np.array(c))
                        try:
                            len(c)
                        except TypeError:
                            c = [c]
                    elif creg == 2:
                        lmbda = 0.001
                        if n_uv > 1:
                            c = 2 * np.dot(np.linalg.pinv(2 * np.dot(kuv.T, kuv) + lmbda * np.eye(n_uv)), np.dot(Y, kuv))
                        else:
                            kuv = kuv[:, np.newaxis]
                            c = 2 * (1 / (2 * np.dot(kuv.T, kuv) + lmbda) * np.dot(Y, kuv))
                elif closs == "cos":
                    from losses_and_gradients import PairwiseCosLoss
                    mycloss = PairwiseCosLoss(X, self.kernel, Y)

                    if creg == 2:
                        def grad_c(c):
                            return mycloss.grad_c(c, u, v) + 0.001 * c

                        def cost_c(c):
                            val = mycloss.cost_uvc(u, v, c) + 0.001 * np.linalg.norm(c) ** 2
                            return val

                        c = self.gd(c, cost_c, grad_c, stepsize=stepsize, max_iter=max_gd_iters, verbosity=verbosity)
                    else:
                        def grad_c(c):
                            return mycloss.grad_c(c, u, v)

                        def cost_c(c):
                            val = mycloss.cost_uvc(u, v, c)
                            return val

                        c = self.prox_gd(c, cost_c, grad_c, gamma)

                # print("c update took ", timedelta(seconds=(time.clock()-t0)))
                # ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ select best results ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤

                if init_loop == 0:
                    self.u = np.copy(u)
                    self.v = np.copy(v)
                    self.c = np.copy(c)
                else:
                    if cost_uvc(u, v, c) < cost_uvc(self.u, self.v, self.c):
                        self.u = np.copy(u)
                        self.v = np.copy(v)
                        self.c = np.copy(c)

        return self.u, self.v, self.c

    def batch_gd(self, x, xbcost, xbgrad, order, batch_size=100, stepsize=10, max_iter=100, verbosity=0):

        # gradient descent on x, with cost xcost and gradient xgrad
        # using batches in each iteration

        updated_once = False

        for ii in range(max_iter):

            elems = order[ii*batch_size:(ii+1)*batch_size]

            gx = xbgrad(x, elems)
            cx = xbcost(x, elems)

            # update
            x_update = x - stepsize * gx
            cx_new = xbcost(x_update, elems)

            if verbosity > 0:
                print("new cost on round", ii, cx_new)

            if np.any(np.isnan(x_update)):
                if verbosity > 0:
                    print("diminishing stepsize (nan encountered)")
                stepsize = 0.1*stepsize
            # add a check: did the loss diminish?
            elif cx < cx_new:
                if verbosity > 0:
                    print("but not updated")
                stepsize = 0.1*stepsize  # if not reduce stepsize; probably overshot it
            else:
                if verbosity > 0:
                    print("updated")
                # update the previous
                diff = np.linalg.norm(x - x_update) / np.linalg.norm(x_update)

                x = np.copy(x_update)
                # cx = cx_new

                # perhaps decrease the stepsize
                if ii >= 1 and updated_once:
                    if diff > prev_diff:
                        if verbosity > 1:
                            print("modifying step size!", 0.1 * stepsize)
                        stepsize = 0.1 * stepsize
                prev_diff = diff

                updated_once = True

                # check for convergence
                if ii > 0 and diff < 1e-4:
                    # print("stuck here")
                    break

                # gx = xgrad(x)

            # also break if stepsize is really tiny
            if ii > 1 and stepsize < 1e-12:
                # print("no, stuck here")
                break

            if verbosity > 0:
                print("cost after round", ii, xbcost(x, elems))

        if verbosity > 0:
            print("final x:")
            print(x)

        return x

    def predict(self, Xt):

        V1t = Xt[0]
        v1inds = Xt[1]
        V2t = Xt[2]
        v2inds = Xt[3]

        if self.standardise:
            V1t = (V1t - self.data_V1_means) / self.data_V1_stds
            V2t = (V2t - self.data_V2_means)
            V2t[:, self.data_V2_std_inds] /= self.data_V2_stds[self.data_V2_std_inds]

        kvec = self.kernel1.kernel(V1t[v1inds, :], np.squeeze(self.u))
        lvec = self.kernel2.kernel(V2t[v2inds, :], np.squeeze(self.v))
        if self.kernel1.zeroone:
            kvec = kvec*2-1
        if self.kernel2.zeroone:
            lvec = lvec*2-1
        klvec = kvec*lvec
        return np.dot(klvec, self.c)
