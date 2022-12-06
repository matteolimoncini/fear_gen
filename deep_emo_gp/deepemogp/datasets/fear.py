import numpy as np
import pandas as pd
from .dataset import Dataset
from scipy import io


class FEAR(Dataset):
    # path of the directory containing physiological signals
    signal_folder = '/media/paolo/Volume/matteo/unimi/tesi_master/code/dataset/amhuse'

    def __init__(self, name='FEAR', signals=None, subjects=None, ):
        super(FEAR, self).__init__(name, signals, subjects, annotations=None)

        if signals is None:  # error!
            raise ValueError("Invalid argument! signals cannot be empty.\n")

        if subjects is None:  # error!
            raise ValueError("Invalid argument! subjects cannot be empty.\n")

        self.signals = signals  # contains the extracted signals data
        self.subjects = subjects  # contains the considered dataset subjects
        # structured as 'person_session'
        # (eg. 1_1 corresponds to subject 1, session 1)
        self.loadData()

    def loadData(self):

        ''' 
        Physiological data in fear gen dataset are structured in
        one npy file, one per each subject.
        '''

        for signal in self.signals:  # loop over all considered signals

            if signal.name is 'EDA':

                for subject in self.subjects:  # loop over all considered subjects

                    person, session = subject.split("_")

                    print(">> Loading %s for subject %s and session %s from dataset %s" % (
                        signal.name, person, session, self.name))

                    # only for test
                    npy_file = '/home/paolo/matteo/matteo/unimi/tesi_master/code/fear_gen/prova.npy'
                    data = np.load(npy_file, allow_pickle=True).item()
                    print('loaded npy file')
                    # save raw data
                    signal.raw.append({'data': iter(data['eda'])})


            elif signal.name is 'HR':

                for subject in self.subjects:  # loop over all considered subjects

                    person, session = subject.split("_")

                    print(">> Loading %s for subject %s and session %s from dataset %s" % (
                        signal.name, person, session, self.name))

                    # only for test
                    npy_file = '/home/paolo/matteo/matteo/unimi/tesi_master/code/fear_gen/prova.npy'
                    data = np.load(npy_file, allow_pickle=True).item()
                    print('loaded npy file')
                    print(data['hr'])
                    # save raw data
                    signal.raw.append({'data': iter(data['hr'])})


            else:
                raise ValueError(
                    "Invalid argument! %s signal is not present in dataset %s.\n" % (signal.name, self.name))
