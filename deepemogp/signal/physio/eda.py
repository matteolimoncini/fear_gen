from ..signal import Signal
from ..utils import utils
from ..utils import cvxEDA
from ...feature_extractor import FE

import biosppy
import neurokit2 as nk

class EDA(Signal):
    """Class to handle the Electro Dermal Activity signal"""

    def __init__(self, feature_ext=FE()):
        super(EDA, self).__init__('EDA', feature_ext)

    def preprocess(self, new_fps=25, show=False, useNeurokit=False):
        '''
        pre process EDA signal to extract phasic component
        '''

        if useNeurokit:
            print(">> Processing using neurokit %s ..." % (self.name))

            for raw in self.raw:  # for each raw data series

                # Filtering
                filtered, _, _ = biosppy.tools.filter_signal(signal=raw['data'],
                                                             ftype='butter',
                                                             band='lowpass',
                                                             order=4,
                                                             frequency=5,
                                                             sampling_rate=raw['fps'])

                # Smoothing
                filtered, _ = biosppy.tools.smoother(signal=filtered,
                                                     kernel='boxzen',
                                                     size=int(0.75 * raw['fps']),
                                                     mirror=True)

                # down-sample data
                tmp_eda = utils.resample(filtered, raw['fps'], new_fps)

                # apply convex optimization from (https://github.com/lciti/cvxEDA)
                # to extract phasic component
                yn = (tmp_eda - tmp_eda.mean()) / tmp_eda.std()

                eda = nk.eda_phasic(yn, sampling_rate=raw['fps'])
                r = eda["EDA_Phasic"].values
                # [r, p, t, l, d, e, obj] = cvxEDA.cvxEDA(yn, 1. / new_fps, options={'show_progress': False})

                print(r)

                # save processed data
                self.processed.append({'data': r, 'fps': new_fps})


        else:
            print(">> Processing %s ..." % (self.name))

            for raw in self.raw:  # for each raw data series

                # Filtering
                filtered, _, _ = biosppy.tools.filter_signal(signal=raw['data'],
                                                             ftype='butter',
                                                             band='lowpass',
                                                             order=4,
                                                             frequency=5,
                                                             sampling_rate=raw['fps'])

                # Smoothing
                filtered, _ = biosppy.tools.smoother(signal=filtered,
                                                     kernel='boxzen',
                                                     size=int(0.75 * raw['fps']),
                                                     mirror=True)

                # down-sample data
                tmp_eda = utils.resample(filtered, raw['fps'], new_fps)

                # apply convex optimization from (https://github.com/lciti/cvxEDA)
                # to extract phasic component
                yn = (tmp_eda - tmp_eda.mean()) / tmp_eda.std()
                [r, p, t, l, d, e, obj] = cvxEDA.cvxEDA(yn, 1. / new_fps, options={'show_progress': False})

                # display the extracted EDA components
                if show:
                    import pylab as pl
                    tm = pl.arange(1., len(tmp_eda) + 1.) / new_fps
                    pl.plot(tm, yn, label='EDA')
                    pl.plot(tm, r, label='Phasic')
                    pl.plot(tm, t, label='Tonic')
                    pl.title('EDA decomposition')
                    pl.legend(loc='best')
                    pl.show()

                # save processed data
                self.processed.append({'data': r, 'fps': new_fps})
