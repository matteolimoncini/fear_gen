import pywt
import pandas as pd
import biosppy
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import idct, dct
import code


class FE(object):
    """Class to handle the feature extractor method"""

    def __init__(self, name='raw', window=(4, 2), skip=0, params={}):
        # name of the feature extractor method
        self.name = name
        # processing window / overlapping length (in seconds)
        # e.g. (4, 2) : 4 seconds window with 2 seconds overlap
        self.window = window
        # number of feature samples to skip from start. useful to align signals
        # when different processing windows are selected
        self.skip = skip
        # additional parameters specific for the extractor
        self.params = params

    def extract_feat(self, signal, show=False):

        print(
            ">> Extracting %s features from %s signal, adopting %s window ..." % (self.name, signal.name, self.window))

        for processed in signal.processed:  # for each processed data series

            chunk_len = processed['fps'] * self.window[0]
            overlap_len = chunk_len - (processed['fps'] * self.window[1])

            # split signal in chunks and apply the related feature extractor
            feature = biosppy.signals.tools.windower(processed['data'], chunk_len, overlap_len, self.apply)['values']

            if show:  # show the reconstruction effectiveness
                rec_signal = []

                step = self.window[0] / (self.window[0] - self.window[1])

                # skip overlapping windows in the reconstruction
                for f in range(0, len(feature), step):
                    rec = self.reconstruct(feature[f])

                    if rec is None:  # in case the adopted feature extractor cannot be reconstructed
                        break

                    rec_signal = np.concatenate([rec_signal, rec])

                if len(rec_signal) > 0:
                    processed = processed['data'][:len(rec_signal)]

                    tm = np.arange(0., len(rec_signal))
                    plt.figure()
                    plt.plot(tm, processed, label='original')
                    plt.plot(tm, rec_signal, label='reconstructed')
                    plt.legend(loc='best')
                    plt.title('Reconstruction of %s signal adopting %s (max err = %.2f)' % (
                    signal.name, self.name, max(abs(processed - rec_signal))))
                    plt.show()

            # eventually skip samples to align different signals
            feature = pd.DataFrame(feature[self.skip:])

            if show:
                if feature.shape[1] > 1:
                    pd.plotting.scatter_matrix(feature, alpha=0.2, figsize=(6, 6), diagonal='kde')
                code.interact(local=locals())
                plt.figure()
                for d in range(feature.shape[1]):
                    plt.plot(feature[d], label='d=' + str(d))
                plt.legend(loc='upper right')
                plt.title('Feature extracted from signal %s' % (signal.name))
                plt.show()

            # concatenate all extracted features
            signal.features = pd.concat([signal.features, feature], ignore_index=True)

    def extract_feat_without_preprocess(self, signal, show=False):

        print(
            ">> Extracting %s features from %s signal, adopting %s window ..." % (self.name, signal.name, self.window))

        for processed in signal.raw:  # for each processed data series

            chunk_len = processed['fps'] * self.window[0]            # 500 * 4 = 2000
            overlap_len = chunk_len - (processed['fps'] * self.window[1]) # 2000 - (500*2) = 1000

            # split signal in chunks and apply the related feature extractor
            feature = biosppy.signals.tools.windower(processed['data'], chunk_len, overlap_len, self.apply)['values']

            if show:  # show the reconstruction effectiveness
                rec_signal = []

                step = int(self.window[0] / (self.window[0] - self.window[1]))

                # skip overlapping windows in the reconstruction
                for f in range(0, len(feature), step):
                    rec = self.reconstruct(feature[f])

                    if rec is None:  # in case the adopted feature extractor cannot be reconstructed
                        break

                    rec_signal = np.concatenate([rec_signal, rec])

                if len(rec_signal) > 0:
                    processed = processed['data'][:len(rec_signal)]

                    tm = np.arange(0., len(rec_signal))
                    plt.figure()
                    plt.plot(tm, processed, label='original')
                    plt.plot(tm, rec_signal, label='reconstructed')
                    plt.legend(loc='best')
                    plt.title('Reconstruction of %s signal adopting %s (max err = %.2f)' % (
                        signal.name, self.name, max(abs(processed - rec_signal))))
                    plt.show()

            # eventually skip samples to align different signals
            feature = pd.DataFrame(feature[self.skip:])

            if show:
                if feature.shape[1] > 1:
                    pd.plotting.scatter_matrix(feature, alpha=0.2, figsize=(6, 6), diagonal='kde')
                code.interact(local=locals())
                plt.figure()
                for d in range(feature.shape[1]):
                    plt.plot(feature[d], label='d=' + str(d))
                plt.legend(loc='upper right')
                plt.title('Feature extracted from signal %s' % (signal.name))
                plt.show()

            # concatenate all extracted features
            signal.features = pd.concat([signal.features, feature], ignore_index=True)

    def apply(self, chunk):
        if self.name == 'raw':
            return chunk

        elif self.name == 'mean':  # mean value
            return np.mean(chunk, axis=0)

        elif self.name == 'wavelet':  # wavelet decomposition
            return self.apply_wavelet(chunk)

        elif self.name == 'dct':  # wavelet decomposition
            return self.apply_dct(chunk)

    def reconstruct(self, feature):

        if self.name == 'raw':
            return feature

        elif self.name == 'mean':  # mean value
            # print "Warning! Cannot reconstruct original data from mean."
            return None

        elif self.name == 'wavelet':  # wavelet decomposition
            return self.reconstruct_wavelet(feature)

        elif self.name == 'dct':  # wavelet decomposition
            return self.reconstruct_dct(feature)

    def apply_dct(self, chunk):
        d = dct(chunk, norm='ortho')
        return d[:20]

    def apply_wavelet(self, chunk):

        # check if wavelet parameters are provided
        if 'w_mother' not in self.params:
            self.params['w_mother'] = 'db3'

        w = pywt.Wavelet(self.params['w_mother'])

        if 'w_maxlev' not in self.params:
            self.params['w_maxlev'] = pywt.dwt_max_level(len(chunk), w.dec_len)

        self.params['w_lenchunk'] = len(chunk)

        # extract only the wavelet approximation coefficient from
        # the latest meaningful level
        return pywt.downcoef('a', chunk, w, level=self.params['w_maxlev'])

    def reconstruct_dct(self, feature):
        feature = np.concatenate([feature, np.zeros(80)])
        return idct(feature, norm='ortho')

    def reconstruct_wavelet(self, feature):
        w = pywt.Wavelet(self.params['w_mother'])

        # try the reconstruction from the considered coefficient
        rec_chunk = pywt.upcoef('a', feature, w, level=self.params['w_maxlev'], take=self.params['w_lenchunk'])

        # smooth the reconstruction
        rec_chunk = biosppy.tools.smoother(rec_chunk)['signal']

        return rec_chunk

    def __str__(self):
        return ("\nFeature extractor\n- type: %s\n- window size: %d\n- window overlap size: %d\n"
                % (self.name, self.window[0], self.window[1]))
