import scipy.signal as sig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_session(model, views, data):
    for v in views:
        # observed inputs variables (annotations)
        X = pd.concat([a.features for a in data.annotations], axis=1).values

        for s in data.signals:
            if s.name == v:
                step = s.feature_ext.window[0] / (s.feature_ext.window[0] - s.feature_ext.window[1])

                Y = s.features.values
                Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
                X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

                pred, var = model.predict(X, view=v)

                rmse = np.sqrt(((np.asarray(pred) - np.asarray(Y)) ** 2).mean())

                show_prediction(pred, var, Y, v)

                pred_rec = []
                Y_rec = []

                for i in range(0, len(pred), step):
                    if s.feature_ext.reconstruct(pred[i, :]) is None:
                        pred_rec.extend(pred[i, :])
                        Y_rec.extend(Y[i, :])
                    else:
                        pred_rec.extend(s.feature_ext.reconstruct(pred[i, :]))
                        Y_rec.extend(s.feature_ext.reconstruct(Y[i, :]))

                # pred_rec = (pred_rec - np.mean(pred_rec)) / np.std(pred_rec)
                # Y_rec = (Y_rec - np.mean(Y_rec)) / np.std(Y_rec)
                rmse = np.sqrt(((np.asarray(pred_rec) - np.asarray(Y_rec)) ** 2).mean())
                cc = np.corrcoef(np.asarray(pred_rec), np.asarray(Y_rec))[0, 1]
                print("%s reconstruction RMSE: %f, CC: %f" % (s.name, rmse, cc))

                f = plt.figure()
                plt.plot(pred_rec, label='reconstruction')
                plt.plot(Y_rec, label='original')
                plt.title('Reconstruction of signal %s' % (s.name))
                plt.legend(loc='best')
                f.show()


def show_prediction(mu, var, Y, signal):
    rmse_pre = np.sqrt(((np.asarray(mu) - np.asarray(Y)) ** 2).mean(axis=0))
    # print rmse_pre
    # print "%s prediction RMSE: %f" % (signal, rmse_pre)
    dim_toshow = np.argmin(rmse_pre)

    mu = mu[:, dim_toshow]
    var = var[:, dim_toshow]
    Y = Y[:, dim_toshow]

    f = plt.figure()

    low = (mu - 2 * np.sqrt(var)).flatten()
    high = (mu + 2 * np.sqrt(var)).flatten()
    plt.fill_between(list(range(len(Y))), low, high, label='mu +- sigma.', alpha=0.2)
    plt.plot(list(range(len(Y))), mu, label='Mu')
    plt.scatter(list(range(len(Y))), Y, label='Data points', s=100)
    plt.title('Predicted %s signal, dim: %d, rmse: %.4f' % (signal, dim_toshow, rmse_pre[dim_toshow]))
    f.show()


def resample(data, from_fps, to_fps):
    new_size = (len(data) / from_fps) * to_fps

    return sig.resample(data, int(new_size))
