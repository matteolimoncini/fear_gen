from ..signal import Signal
from ..utils import utils
from ...feature_extractor import FE

import biosppy


class SKT(Signal):
    """Class to handle the Skin Temperature signal"""

    def __init__(self, feature_ext=FE()):
        super(SKT, self).__init__('SKT', feature_ext)

    def preprocess(self, new_fps=25, show=False):
        '''
        pre process SKT signal
        '''

        fo = 2  # filter order
        fc = 8  # filter cut frequency
        ft = 'lowpass'  # filter type

        print(">> Processing %s ..." % (self.name))

        for raw in self.raw:  # for each raw data series

            '''
            Barnea, O., & Shusterman, V. (1995, September). Analysis of skin-temperature variability compared to variability of blood pressure and heart rate. In Engineering in Medicine and Biology Society, 1995., IEEE 17th Annual Conference (Vol. 2, pp. 1027-1028). IEEE.
            '''
            nyq = 0.5 * raw['fps']

            # Filtering
            filtered, _, _ = biosppy.tools.filter_signal(signal=raw['data'],
                                                         ftype='butter',
                                                         band='lowpass',
                                                         order=fo,
                                                         frequency=fc / nyq,
                                                         sampling_rate=raw['fps'])

            if show:
                import pylab as pl
                tm = pl.arange(1., len(raw['data']) + 1.) / new_fps
                pl.plot(tm, raw['data'], label='raw SKT')
                pl.plot(tm, filtered, label='filtered SKT')
                pl.title('SKT filtering')
                pl.legend(loc='best')
                pl.show()

            # Down-sample data
            tmp_skt = utils.resample(filtered, raw['fps'], new_fps)

            self.processed.append({'data': tmp_skt, 'fps': new_fps})
