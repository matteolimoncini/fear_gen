from matplotlib import pylab as plt
import numpy as np
# from: https://github.com/SheffieldML/GPy
import GPy

# from: https://github.com/SheffieldML/PyDeepGP
import deepgp

import scipy.io

plt.ioff()


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


mat = scipy.io.loadmat('data/data.mat')

X = mat['annotations']
Y = mat['Yall'][0, 0]

N = X.shape[0]
Ntr = 70
# X = np.linspace(0,1,N)
Xmean = X.mean()
Xstd = X.std()

X -= Xmean
X /= Xstd

# Y = []
# fac=4
# for i in range(N):
#    tmp = np.sin(fac*X[i]) * np.cos(fac*X[i]*X[i])
#    Y.append(tmp)

# Y = np.array(Y)[:,None]
# X = X[:,None]

Yorig = Y.copy()

error_GP = []
error_DGP = []

# Perform 5 random trials
for j in range(1):
    print(('# Running repeat: ', str(j)))
    Y = Yorig.copy()
    # Y+= np.random.randn(Y.shape[0])[:,None]* 0.05

    # split in train and test
    perm = np.random.permutation(N)
    index_tr = np.sort(perm[0:Ntr])
    index_ts = np.sort(perm[Ntr:])
    Xtr = X[index_tr, :]
    Ytr = Y[index_tr, :]
    Xts = X[index_ts, :]
    Yts = Y[index_ts, :]

    # GP baseline
    mGP = GPy.models.SparseGPRegression(X=Xtr, Y=Ytr, num_inducing=Ntr)
    mGP.optimize(max_iters=4000, messages=1)
    pred_GP = mGP.predict(Xts)[0]

    k1 = GPy.kern.RBF(6, 1., 2.)
    k2 = GPy.kern.Matern32(6, 0.5, 0.2)
    k_prod = k1 * k2

    # DGP baseline
    m = deepgp.DeepGP([Ytr.shape[1], 6, Xtr.shape[1]], Ytr, X=Xtr, kernels=[k_prod, GPy.kern.RBF(X.shape[1])],
                      num_inducing=Ntr, back_constraint=False)
    # For so simple 1D data, we initialize the middle layer (latent space) to
    # be the input.
    m.layer_1.X.mean = Xtr.copy()
    # We can initialize and then fix the middle layer inducing points to also be the input.
    # Another strategy would be to (also/or) do that in the top layer.
    # m.obslayer.Z[:] = Xtr[:].copy()
    # m.obslayer.Z.fix()
    m.layer_1.Z[:] = Xtr[:].copy()
    m.layer_1.Z.fix()
    # Here we initialize such that Signal to noise ratio is high.
    for i in range(len(m.layers)):
        if i == 0:
            m.layers[i].Gaussian_noise.variance = m.layers[i].Y.var() * 0.01
        else:
            m.layers[i].Gaussian_noise.variance = m.layers[i].Y.mean.var() * 0.005
        # Fix noise for a few iters, so that it avoids the trivial local
        # minimum to only learn noise.
        m.layers[i].Gaussian_noise.variance.fix()

    m.optimize(max_iters=200, messages=1)
    # Now unfix noise and learn normally.
    for i in range(len(m.layers)):
        m.layers[i].Gaussian_noise.variance.unfix()
    m.optimize(max_iters=4000, messages=1)

    pred = m.predict(Xts)[0]

    Xtmp = np.linspace(0, 1, Xts.shape[0])
    # plt.plot(Xts[:,0], pred_GP[:,0], 'x-r', label='GP')
    # plt.plot(Xts[:,0], pred[:,0], 'o-b', label='DGP')
    # plt.plot(Xts[:,0], Yts[:,0], '+k--', label='True')
    for d in range(Yts.shape[1]):
        plt.plot(Xtmp, pred_GP[:, d], 'x-r', label='GP')
        plt.plot(Xtmp, pred[:, d], 'o-b', label='DGP')
        plt.plot(Xtmp, Yts[:, d], '+k--', label='True')
        plt.legend()
        plt.show()

    error_DGP.append(rmse(pred, Yts))
    error_GP.append(rmse(pred_GP, Yts))

error_GP = np.array(error_GP)
error_DGP = np.array(error_DGP)
print(('# Error GP:  ', error_GP.mean(), ' with std: ', error_GP.std()))
print(('# Error DGP: ', error_DGP.mean(), ' with std: ', error_DGP.std()))
