from ..signal import Signal
from ..utils import utils
from ...feature_extractor import FE

import pandas as pd
import numpy as np
import biosppy
from scipy import interpolate


class ECG(Signal):
    """Class to handle the Electro CardioGram signal"""

    def __init__(self, feature_ext=FE()):
        super(ECG, self).__init__('ECG', feature_ext)

    def preprocess(self, new_fps=25, show=False):
        '''
        pre process ECG signal to extract Heart Rate signal
        '''

        print(">> Processing %s ..." % (self.name))

        for raw in self.raw:  # for each raw data series

            # Extract Heart Rate values
            results = biosppy.signals.ecg.ecg(raw['data'], raw['fps'], show)
            t_raw = results[0]
            t_hr = results[5]
            hr = results[6]

            # Interpolate HR values to bring signal back to original size
            f = interpolate.interp1d(t_hr, hr, bounds_error=False, fill_value="extrapolate")
            tmp_HR = f(t_raw)

            fps = int(len(tmp_HR) / t_raw[-1])

            self.processed.append({'data': tmp_HR, 'fps': fps})
