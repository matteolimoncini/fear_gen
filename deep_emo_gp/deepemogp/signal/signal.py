import pandas as pd


class Signal(object):
    """Class to handle a general signal"""

    def __init__(self, name, feature_ext):
        # name of the physiological signal
        self.name = name
        # contains the raw data and relative sampling rate
        # eg. {'data': [], 'fps': []}
        self.raw = []
        # contains the processed data and relative sampling rate
        # eg. {'data': [], 'fps': []}
        self.processed = []
        # contains the extracted features
        self.features = pd.DataFrame()
        # instance of the feature extraction method
        self.feature_ext = feature_ext

    def __str__(self):
        raw_size = []
        for s in self.raw:
            raw_size.append(s['data'].shape[0])

        proc_size = []
        for s in self.processed:
            proc_size.append(s['data'].shape[0])

        return (
                    "\nSignal\n- type: %s\n- raw series: %s\n- raw size: %s\n- processed size: %s\n- feature type: %s\n- features size: %s\n"
                    % (self.name, len(self.raw), raw_size, proc_size, self.feature_ext.name, self.features.shape))
