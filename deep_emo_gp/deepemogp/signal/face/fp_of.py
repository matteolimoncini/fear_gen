from ..signal import Signal
from ..utils import utils
from ...feature_extractor import FE
from ...feature_extractor.candide import candide
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


class FP_OF(Signal):
    """Class to handle the facial fiducial points extracted via Openface"""

    def __init__(self, feature_ext=FE()):
        super(FP_OF, self).__init__('FP_OF', feature_ext)

    def preprocess(self, new_fps=25, show=False):
        '''
        Visuomotor mapping from landmarks to candide motor parameters
        (l -> m)
        '''

        print(">> Processing %s ..." % (self.name))

        for raw in self.raw:  # for each raw data series
            data = raw['data']

            x = data.filter(regex='x_*', axis=1)
            y = data.filter(regex='y_*', axis=1)

            if show:  # show raw landmarks
                fig, ax = plt.subplots()
                ax.invert_yaxis()
                ax.set_aspect('equal')

                sc = plt.scatter(x.loc[0, :], y.loc[0, :])

                axframe = plt.axes([0.1, 0.01, 0.8, 0.05])
                sframe = Slider(axframe, 'Frame', 0, len(data) - 1, valinit=0)

                def update(val):
                    frame = int(sframe.val)
                    sc.set_offsets(np.c_[x.loc[frame, :], y.loc[frame, :]])
                    ax.set_title("Fiducial Points - frame %d/%d" % (frame, len(data) - 1))
                    fig.canvas.draw_idle()

                sframe.on_changed(update)

                plt.show()

            # realise the AUV weights extraction
            w = candide.mapToCandide(np.c_[x, y], 'OF', show)

            # resample data
            w = utils.resample(w, raw['fps'], new_fps)

            if show:
                plt.plot(w)
                plt.title('Action Unit weights')
                plt.show()

            self.processed.append({'data': w, 'fps': new_fps})
