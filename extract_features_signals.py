import numpy as np
import pandas as pd

import extract_correct_csv
from deepemogp import datasets as datasets
from deepemogp import feature_extractor
from deepemogp.signal import behavior as behavior
from deepemogp.signal import physio as physio
import warnings
import os

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

all_subjects = extract_correct_csv.extract_only_valid_subject()
all_subjects.remove(11)

list_less_trials = [13, 15, 16, 17, 23, 26, 27, 28, 31, 32, 33]
path = "data/features_4_2"
for i in all_subjects:
    TRIAL = 160
    if i in list_less_trials:
        TRIAL = 159

    check_path_eda = 'data/features_4_2/eda/' + str(i) + '.csv'
    check_path_hr = 'data/features_4_2/hr/' + str(i) + '.csv'
    check_path_pupil = 'data/features_4_2/pupil/' + str(i) + '.csv'
    if (os.path.exists(check_path_hr)) & (os.path.exists(check_path_eda)) & (os.path.exists(check_path_pupil)):
        continue

    f2 = feature_extractor.FE('wavelet', window=(4, 2))
    f3 = feature_extractor.FE('wavelet', window=(3, 1))

    show = False

    # definition of the physiological signals to be extracted
    eda_ = physio.EDA(f2)
    hr_ = physio.ECG(f2)
    pupil_ = behavior.PUPIL(f3)

    d = datasets.FEAR(signals={hr_, pupil_, eda_}, subjects={str(i)})

    try:
        for s in d.signals:
            # preprocess ...
            if s.name == 'EDA':
                s.preprocess(show=show, new_fps=500)
                s.feature_ext.extract_feat(s, show=show)
            else:
                if s.name == 'ECG':
                    list_hr_test = s.raw[0]['data']
                    s.preprocess(show=show, useneurokit=False)
                    s.feature_ext.extract_feat(s, show=show)


                else:
                    s.feature_ext.extract_feat_without_preprocess(s, show=show)

            # add feature extraction for eda before preprocessing
            # ... and extract features from each signal type

        for sig in d.signals:
            if sig.name == 'EDA':
                eda_data = sig.features
            if sig.name == 'ECG':
                hr_data = sig.features
            if sig.name == 'PUPIL':
                pupil_data = sig.features

        hr = np.array(hr_data)
        hr = hr.reshape((TRIAL, int(hr.shape[0] / TRIAL * hr.shape[1])))
        pd.DataFrame(hr).to_csv(path + '/hr/' + str(i) + '.csv', index=False)

        eda = np.array(eda_data)
        eda = eda.reshape((TRIAL, int(eda.shape[0] / TRIAL * eda.shape[1])))
        pd.DataFrame(eda).to_csv(path + '/eda/' + str(i) + '.csv', index=False)

        pupil = np.array(pupil_data)
        pupil = pupil.reshape((TRIAL, int(pupil.shape[0] / TRIAL * pupil.shape[1])))
        pd.DataFrame(pupil).to_csv(path + '/pupil/' + str(i) + '.csv', index=False)
    except ArithmeticError:
        print(f'division by zero from subject {str(i)}')
