from ..signal import Signal
from ..utils import utils
from ...feature_extractor import FE
from ...feature_extractor.candide import candide
import matplotlib.pyplot as plt


class AU_OF(Signal):
    """Class to handle the Action Units extracted via Openface"""

    def __init__(self, feature_ext=FE()):
        super(AU_OF, self).__init__('AU_OF', feature_ext)

    def preprocess(self, new_fps=25, show=False):
        '''
        conversion of motor parameters, from Ekman AU to Candide AUV
        (m_e -> m_c)
        '''

        print(">> Processing %s ..." % (self.name))

        for raw in self.raw:  # for each raw data series

            fields = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', \
                      'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', \
                      'AU26_r', 'AU45_r']

            w = raw['data']

            if show:  # show raw action units
                plt.plot(w)
                plt.legend(fields)
                plt.title('Raw Action Units')
                plt.show()

            # realise the AUV conversion
            w = candide.convertAU(w, 'OF', show)

            # down-sample data
            w = utils.resample(w, raw['fps'], new_fps)

            self.processed.append({'data': w, 'fps': new_fps})
