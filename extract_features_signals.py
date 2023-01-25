import numpy as np
import pandas as pd

import extract_correct_csv
from deepemogp import datasets as datasets
from deepemogp import feature_extractor
from deepemogp.signal import behavior as behavior
from deepemogp.signal import physio as physio
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

prova_3_subj = extract_correct_csv.extract_only_valid_subject()
prova_3_subj.remove(49)
TRIAL = 160
path = "/home/paolo/matteo/matteo/unimi/tesi_master/code/fear_gen/data_fake/features_4_2"

for i in prova_3_subj:

    f2 = feature_extractor.FE('wavelet', window=(4, 2))
    f3 = feature_extractor.FE('mean', window=(1, 0))

    show = False

    # definition of the physiological signals to be extracted
    eda_ = physio.EDA(f2)
    hr_ = physio.ECG(f2)
    # pupil_ = behavior.PUPIL(f3)

    d = datasets.FEAR(signals={hr_, eda_}, subjects={str(i)})  #pupil_,

    for s in d.signals:
        # preprocess ...
        if s.name == 'EDA':
            pass
            # s.preprocess(show=show, new_fps=500)
            #s.feature_ext.extract_feat(s, show=show)
        else:
            if s.name == 'ECG':
                # list_hr_test = s.raw[0]['data']
                s.preprocess(show=show, useneurokit=True)
                s.feature_ext.extract_feat(s, show=show)

            else:
                s.feature_ext.extract_feat_without_preprocess(s, show=show)

        # add feature extraction for eda before preprocessing

        # ... and extract features from each signal type

    for sig in d.signals:
        if sig.name == 'EDA':
            pass
            # eda_data = sig.features
        if sig.name == 'ECG':
            hr_data = sig.features
        if sig.name == 'PUPIL':
            pass
            # pupil_data = sig.features

    hr = np.array(hr_data)
    hr = hr.reshape((TRIAL, int(hr.shape[0] / TRIAL * hr.shape[1])))
    pd.DataFrame(hr).to_csv(path + '/hr/' + str(i) + '.csv', index=False)

    # eda = np.array(eda_data)
    # eda = eda.reshape((TRIAL, int(eda.shape[0] / TRIAL * eda.shape[1])))
    # pd.DataFrame(eda).to_csv(path + '/eda/' + str(i) + '.csv', index=False)

    # pupil = np.array(pupil_data)
    # pupil = pupil.reshape((TRIAL, int(pupil.shape[0] / TRIAL * pupil.shape[1])))
    #pd.DataFrame(pupil).to_csv(path + '/pupil/' + str(i) + '.csv', index=False)
