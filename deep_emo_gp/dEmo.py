import deepemogp.feature_extractor as feature_extractor
import deepemogp.signal.physio as physio
import deepemogp.signal.annotation as annotation
import deepemogp.datasets as datasets

# from: https://github.com/SheffieldML/GPy
import GPy
# from: https://github.com/SheffieldML/PyDeepGP
import deepgp

import pandas as pd
import numpy as np

show = True
learn = True

# definition of the feature extractors to be used later
f1 = feature_extractor.FE('wavelet')
f2 = feature_extractor.FE('wavelet', params={'w_mother': 'db4'})
f3 = feature_extractor.FE('mean', skip=2)

# definition of the physiological signals to be extracted
eda = physio.SKT(f1)
ecg = physio.HR(f2)

# definition of the emotional annotation to be extracted
va = annotation.VA('valence', f3)
ar = annotation.VA('arousal', f3)

# extraction of the desired data from the dataset
d = datasets.AMHUSE(signals={eda, ecg}, subjects={'1_1'}, annotations={va, ar})
print(d)

for s in d.signals:
    # preprocess ...
    s.preprocess(show=show)

    # ... and extract features from each signal type   
    s.feature_ext.extract_feat(s, show=show)

    print(s)

for a in d.annotations:
    # preprocess ...
    a.preprocess(ewe=False, show=show)

    # ... and extract features for each annotation type
    a.feature_ext.extract_feat(a, show=show)
    print(a)

# learn model

# observed inputs variables (annotations)
X = pd.concat([a.features for a in d.annotations], axis=1).values
# observed output variables (measured signals)
Y = [s.features.values for s in d.signals]
# Number of latent dimensions (single hidden layer, since the top layer is observed)
Q = 5

k1 = GPy.kern.RBF(Q, ARD=True) + GPy.kern.Bias(Q)
k2 = GPy.kern.RBF(Q, ARD=True) + GPy.kern.Bias(Q)

# k1 = GPy.kern.RBF(Y[0].shape[1], ARD=True) + GPy.kern.Bias(Y[0].shape[1])
# k2 = GPy.kern.RBF(Y[1].shape[1], ARD=True) + GPy.kern.Bias(Y[1].shape[1])

k3 = GPy.kern.RBF(X.shape[1], ARD=False) + GPy.kern.Bias(X.shape[1])

Ntr = 100

# deepGP
m = deepgp.DeepGP([[Y[0].shape[1], Y[1].shape[1]], Q, X.shape[1]], Y, X=X, kernels=[[k1, k2], k3], num_inducing=Ntr,
                  back_constraint=False)

if learn:
    # --------- Optimization ----------#
    # Make sure initial noise variance gives a reasonable signal to noise ratio.
    # Fix to that value for a few iterations to avoid early local minima
    output_var = m.layers[0].view_0.Y.var()
    m.layers[0].view_0.Gaussian_noise.variance = output_var * 0.01
    m.layers[0].view_0.Gaussian_noise.variance.fix()

    output_var = m.layers[0].view_1.Y.var()
    m.layers[0].view_1.Gaussian_noise.variance = output_var * 0.01
    m.layers[0].view_1.Gaussian_noise.variance.fix()

    output_var = m.layers[1].Y.mean.var()
    m.layers[1].Gaussian_noise.variance = output_var * 0.01
    m.layers[1].Gaussian_noise.variance.fix()

    m.optimize(max_iters=800, messages=True)

    m.layers[0].view_0.Gaussian_noise.variance.unfix()
    m.layers[0].view_1.Gaussian_noise.variance.unfix()
    m.layers[1].Gaussian_noise.variance.unfix()
    m.optimize(max_iters=1500, messages=True)

    print(m)

    # Saving a model
    np.save('model_save.npy', m.param_array)

else:

    m.update_model(False)  # do not call the underlying expensive algebra on load
    m.initialize_parameter()  # Initialize the parameters (connect the parameters up)
    m[:] = np.load('model_save.npy')  # Load the parameters
    m.update_model(True)  # Call the algebra only once

from pylab import *

m.obslayer.view_0.kern.plot_ARD('rbf')
plt.show()
m.obslayer.view_1.kern.plot_ARD('rbf')
plt.show()
