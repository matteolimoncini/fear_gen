from ..signal import Signal
from ..utils import utils
from ...feature_extractor import FE


class HR(Signal):
    """Class to handle the Heart Rate signal"""

    def __init__(self, feature_ext=FE()):
        super(HR, self).__init__('HR', feature_ext)

    def preprocess(self, new_fps=25, show=False):
        '''
        pre process HR signal
        '''

        print(">> Processing %s ..." % (self.name))

        for raw in self.raw:  # for each raw data series
            # Heart Rate can be already considered a feature

            # down-sample data
            tmp_HR = utils.resample(raw['data'], raw['fps'], new_fps)

            self.processed.append({'data': tmp_HR, 'fps': new_fps})
