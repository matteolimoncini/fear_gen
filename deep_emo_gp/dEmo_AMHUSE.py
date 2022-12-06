import deepemogp.feature_extractor as feature_extractor
import deepemogp.signal.physio as physio
import deepemogp.signal.face as face
import deepemogp.signal.annotation as annotation
import deepemogp.datasets as datasets
# from: https://github.com/SheffieldML/GPy
import GPy
# from: https://github.com/SheffieldML/PyDeepGP
import deepgp

import pandas as pd
import numpy as np
import code

show_pre, show_post = False, True

loadData = False
learnModel = False

if loadData:

    # definition of the feature extractors to be used later
    f1 = feature_extractor.FE('wavelet')
    f2 = feature_extractor.FE('wavelet')
    f3 = feature_extractor.FE('wavelet')
    f4 = feature_extractor.FE('mean')

    # definition of the physiological signals to be extracted
    skt = physio.SKT(f1)
    hr = physio.HR(f2)
    of = physio.EDA(f3)

    # definition of the emotional annotation to be extracted
    va = annotation.VA('valence', f4)
    ar = annotation.VA('arousal', f4)

    # extraction of the desired data from the dataset
    d = datasets.AMHUSE(signals=[skt, hr, of], subjects=['10_1', '10_3', '10_4'], annotations=[va, ar])
    print(d)

    for s in d.signals:
        # preprocess ...
        s.preprocess(show=show_pre)

        # ... and extract features from each signal type
        s.feature_ext.extract_feat(s, show=show_post)

        print(s)

    for a in d.annotations:
        # preprocess ...
        a.preprocess(ewe=False, show=show_pre)

        # ... and extract features for each annotation type
        a.feature_ext.extract_feat(a, show=show_post)
        print(a)

    # Saving a model
    np.save('data/data/data_' + d.name + '_' + '_'.join(d.subjects) + '.npy', d)

else:

    d = np.load('data/data/data_' + dataset + '_' + '_'.join(subjects) + '.npy')  # Load the data
    d = d.item()

# create model

# observed inputs variables (annotations)
X = pd.concat([a.features for a in d.annotations], axis=1).values
# observed output variables (measured signals)
Y = [s.features.values for s in d.signals]
s_names = [s.name for s in d.signals]

# Xmean = X.mean(axis=0)
# Xstd = X.std(axis=0)
# X-=Xmean
# X/=Xstd

N = len(X)

# Number of latent dimensions (single hidden layer, since the top layer is observed)
d_views = [signal.shape[1] for signal in Y]

Q = [3, 2, X.shape[1]]

k_views = [GPy.kern.RBF(Q[0], ARD=True) + GPy.kern.Bias(Q[0])] * len(Y)

k4 = GPy.kern.RBF(Q[1], ARD=False) + GPy.kern.Bias(Q[1])
k5 = GPy.kern.RBF(Q[2]) + GPy.kern.Bias(Q[2])

encoder_dims = [[300], [150]]

# deepGP
m = deepgp.DeepGP([d_views, Q[0], Q[1], Q[2]], Y=Y,
                  kernels=[k_views, k4, k5],
                  num_inducing=N,
                  back_constraint=False,
                  encoder_dims=encoder_dims,
                  view_names=s_names,
                  inits='PCA_single')

if learnModel:
    # We can initialize and then fix the middle layer inducing points to also be the input.
    # Another strategy would be to (also/or) do that in the top layer.
    m.layers[2].Z[:] = X[:].copy()
    m.layers[2].Z.fix()

    for i in range(len(m.layers)):
        if i == 0:
            for v in range(len(m.layers[i].views)):
                m.layers[i].views[v].Gaussian_noise.variance = m.layers[i].views[v].Y.var() * 0.01
                m.layers[i].views[v].Gaussian_noise.variance.fix()
        else:
            m.layers[i].Gaussian_noise.variance = m.layers[i].Y.mean.var() * 0.01
            # Fix noise for a few iters, so that it avoids the trivial local minimum to only learn noise.
            m.layers[i].Gaussian_noise.variance.fix()

    m.optimize(max_iters=200, messages=1)
    deepgp.util.check_snr(m)

    # Now unfix noise and learn normally.
    for i in range(len(m.layers)):
        if i == 0:
            for v in range(len(m.layers[i].views)):
                m.layers[i].views[v].Gaussian_noise.variance.unfix()
        else:
            m.layers[i].Gaussian_noise.variance.unfix()

    m.optimize(max_iters=4000, messages=1)
    deepgp.util.check_snr(m)

    print(m)

    # Saving a model
    np.save('data/models/model_' + d.name + '_'.join(d.subjects) + '.npy', m.param_array)

else:

    m.update_model(False)  # do not call the underlying expensive algebra on load
    m.initialize_parameter()  # Initialize the parameters (connect the parameters up)
    m[:] = np.load('data/models/data_' + d.name + '_'.join(d.subjects) + '.npy')  # Load the parameters
    m.update_model(True)  # Call the algebra only once

import matplotlib.pyplot as plt

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)
m.obslayer.views[0].kern.plot_ARD('rbf', ax=ax1, title=m.obslayer.views[0].name)
m.obslayer.views[1].kern.plot_ARD('rbf', ax=ax2, title=m.obslayer.views[1].name)
m.obslayer.views[2].kern.plot_ARD('rbf', ax=ax3, title=m.obslayer.views[2].name)
f.show()

lbls = np.zeros(N, dtype=int)

plt.subplot(131)
deepgp.util.visualize_DGP(m, lbls, layer=0, dims=[0, 1]);
plt.title('Layer 0')
plt.subplot(132)
deepgp.util.visualize_DGP(m, lbls, layer=1, dims=[0, 1]);
plt.title('Layer 1')
plt.subplot(133)
# deepgp.util.visualize_DGP(m, lbls, layer=2, dims=[0,1]);
# plt.title('Layer 2')
plt.show()

# Generate session - Experiment 1
from deepemogp.signal.utils import utils

utils.generate_session(m, s_names, d)

# G = m.build_pydot()
# G.write_png('example_hierarchy_layout.png')
