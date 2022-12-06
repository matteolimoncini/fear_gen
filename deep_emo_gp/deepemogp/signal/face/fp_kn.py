from ..signal import Signal
from ..utils import utils
from ...feature_extractor import FE
from ...feature_extractor.candide import candide
import pylab as plt


class FP_KN(Signal):
    """Class to handle the facial fiducial points extracted with Kinect"""

    def __init__(self, feature_ext=FE()):
        super(FP_KN, self).__init__('FP_KN', feature_ext)

    def preprocess(self, new_fps=25, show=False):
# TODO
