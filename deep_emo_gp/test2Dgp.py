from matplotlib import pylab as plt
import numpy as np
# from: https://github.com/SheffieldML/GPy
import GPy

# from: https://github.com/SheffieldML/PyDeepGP
import deepgp

import scipy.io

plt.ioff()

np.random.seed(0)
n = 60  # number of data
d = 1  # number of dimensions
# dependent variable is linearly spaced.
X = np.linspace(-np.pi * 2., np.pi * 2., n)[:, None]
# response variable is step function
Y = np.sin(X) + np.random.randn(n, 1) * 0.02
# where to plot the model predictions
Xtest = np.linspace(-np.pi * 2., np.pi * 2, 300)[:, None]

k1 = GPy.kern.RBF(1)
k2 = GPy.kern.Matern32(1)
k_prod = k1 * k2

Ntr = Y.shape[0];

# GP
mGP = GPy.models.SparseGPRegression(X=X, Y=Y, num_inducing=Ntr)
mGP.optimize('bfgs', max_iters=4000, messages=1)
pred_GP = mGP.predict(Xtest)[0]

# deepGP
m = deepgp.DeepGP([Y.shape[1], 1, X.shape[1]], Y, X=X, kernels=[k1, GPy.kern.RBF(X.shape[1])], num_inducing=Ntr,
                  back_constraint=False)
# for i in range(len(m.layers)):
#     if i == 0:
#         m.layers[i].Gaussian_noise.variance = m.layers[i].Y.var() * 0.01
#     else:
#         m.layers[i].Gaussian_noise.variance = m.layers[i].Y.mean.var() * 0.005
#     # Fix noise for a few iters, so that it avoids the trivial local
#     # minimum to only learn noise.
#     m.layers[i].Gaussian_noise.variance.fix()

# m.optimize(max_iters=200, messages=1)
# # Now unfix noise and learn normally.
# for i in range(len(m.layers)):
#     m.layers[i].Gaussian_noise.variance.unfix()
m.optimize(max_iters=4000, messages=1)

pred = m.predict(Xtest)[0]

plt.plot(Xtest, pred, '+k--', label='Test dgp')
plt.plot(Xtest, pred_GP, '+b--', label='Test gp')

plt.plot(X, Y, 'x-r', label='Train')

plt.legend()
plt.show()
