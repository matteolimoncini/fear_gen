from ..signal import Signal
from ..utils import utils
from ...feature_extractor import FE
import neurokit2 as nk


class HR(Signal):
    """Class to handle the Heart Rate signal"""

    def __init__(self, feature_ext=FE()):
        super(HR, self).__init__('HR', feature_ext)

    def preprocess(self, new_fps=25, show=False, useneurokit=False):
        '''
        pre process HR signal
        '''

        if useneurokit:
            print(">> Processing %s ... using neurokit" % (self.name))
            df, info = nk.ecg_process(list(self.raw[0]['data']))
            tmp_HR = df['ECG_Rate']
            new_fps = 500

        else:
            print(">> Processing %s ... without neurokit only resample" % (self.name))

            for raw in self.raw:  # for each raw data series
                # Heart Rate can be already considered a feature

                # down-sample data
                tmp_HR = utils.resample(raw['data'], raw['fps'], new_fps)

        self.processed.append({'data': tmp_HR, 'fps': new_fps})
