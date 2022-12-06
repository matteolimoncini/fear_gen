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

show = True
loadData = True
learnModel = True

if loadData:

    # definition of the feature extractors to be used later
    f1 = feature_extractor.FE('wavelet')
    f2 = feature_extractor.FE('wavelet')
    f3 = feature_extractor.FE('mean')

    # definition of the physiological signals to be extracted
    skt = physio.SKT(f1)
    hr = physio.HR(f2)
    of = face.FP_OF(f3)

    # definition of the emotional annotation to be extracted
    va = annotation.VA('valence', f3)
    ar = annotation.VA('arousal', f3)

    # extraction of the desired data from the dataset
    d = datasets.AMHUSE(signals={skt, hr, of}, subjects={'1_1', '1_3', '1_4'}, annotations={va, ar})
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

    # Saving a model
    np.save('data.npy', d)

else:

    d = np.load('data.npy')  # Load the data
    d = d.item()

# create model
selected_view = 'HR'

# observed inputs variables (annotations)
X = pd.concat([a.features for a in d.annotations], axis=1).values
# observed output variables (measured signals)
Y = [s.features.values for s in d.signals if s.name == selected_view][0]

Xmean = X.mean()
Xstd = X.std()

# X-=Xmean
# X/=Xstd

N = len(X)

# Number of latent dimensions (single hidden layer, since the top layer is observed)
Q = [Y.shape[1], 6, X.shape[1]]

k1 = GPy.kern.RBF(Q[1], ARD=True) + GPy.kern.Bias(Q[1]) + GPy.kern.White(Q[1])

k2 = GPy.kern.Linear(Q[2], ARD=False)

# Dimensions of the MLP back-constraint if set to true
encoder_dims = [[300], [150]]

# inits = ['PPCA', 'PPCA' , X]

# deepGP
# m = deepgp.DeepGP([d1, Q1, X.shape[1]], Y=Ytr, X=Xtr, 
m = deepgp.DeepGP(Q, Y=Y, X=X,
                  kernels=[k1, k2],
                  num_inducing=N,
                  back_constraint=True,
                  encoder_dims=encoder_dims)

if learnModel:
    # --------- Optimization ----------#
    for i in range(len(m.layers)):
        # Make sure initial noise variance gives a reasonable signal to noise ratio.
        # Fix to that value for a few iterations to avoid early local minima
        output_var = m.layers[i].Y.var() if i == 0 else m.layers[i].Y.mean.var()
        m.layers[i].Gaussian_noise.variance = output_var * 0.1
        m.layers[i].Gaussian_noise.variance.fix()

    m.optimize(max_iters=1000, messages=1)
    deepgp.util.check_snr(m)

    # Now unfix noise and learn normally.
    for i in range(len(m.layers)):
        m.layers[i].Gaussian_noise.variance.unfix()

    # m.optimize_restarts(parallel=False, messages=1, robust=True, num_restarts=3, max_iters=250)
    m.optimize(max_iters=10000, messages=1)
    deepgp.util.check_snr(m)

    print(m)

    # Saving a model
    np.save('model_save.npy', m.param_array)

else:

    m.update_model(False)  # do not call the underlying expensive algebra on load
    m.initialize_parameter()  # Initialize the parameters (connect the parameters up)
    m[:] = np.load('model_save.npy')  # Load the parameters
    m.update_model(True)  # Call the algebra only once

# Show ARD results
import matplotlib.pyplot as plt

m.obslayer.kern.plot_ARD('rbf')
plt.show()

# Generate session - Experiment 1
from deepemogp.signal.utils import utils

utils.generate_session(m, [selected_view], d)

# pred = m.predict(Xts)[0]

# rmse = np.sqrt(((pred.flatten() - Yts.flatten()) ** 2).mean())

# print "prediction RMSE: %f" % (rmse)


# import matplotlib.pyplot as plt
# f = plt.figure(); plt.plot(pred.T, 'x-');  plt.plot(Yts.T, 'o'); f.show();
