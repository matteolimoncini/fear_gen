import numpy as np
import pandas as pd
from .dataset import Dataset
from scipy import io

class FEAR(Dataset):
    # path of the directory containing physiological signals
    signal_folder = '/home/paolo/matteo/matteo/unimi/tesi_master/code/fear_gen/signal'

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

                    csv_file = self.signal_folder + '/eda_csv/' + person + '_eda.csv'
                    df = pd.read_csv(csv_file)
                    session = int(session) - 1
                    # pupil_.loc[i, :].values.flatten().tolist()
                    values = df.loc[session, :].values.flatten().tolist()
                    fps = int(len(values)/6)
                    signal.raw.append({'data': values, 'fps': fps})
                    print(signal.raw)

            elif signal.name is 'PUPIL':

                for subject in self.subjects:  # loop over all considered subjects

                    person, session = subject.split("_")

                    print(">> Loading %s for subject %s and session %s from dataset %s" % (
                        signal.name, person, session, self.name))

                    csv_file = self.signal_folder + '/pupil_csv/' + person + '_pupil.csv'
                    df = pd.read_csv(csv_file)
                    session = int(session) - 1
                    # pupil_.loc[i, :].values.flatten().tolist()
                    values = df.loc[session, :].values.flatten().tolist()
                    fps = int(len(values)/5)
                    signal.raw.append({'data': values, 'fps':fps})
                    print(signal.raw)

            elif signal.name is 'HR':

                for subject in self.subjects:  # loop over all considered subjects

                    person, session = subject.split("_")

                    print(">> Loading %s for subject %s and session %s from dataset %s" % (
                        signal.name, person, session, self.name))

                    csv_file = self.signal_folder + '/hr_csv/' + person + '_hr.csv'
                    df = pd.read_csv(csv_file)
                    session = int(session) - 1
                    #pupil_.loc[i, :].values.flatten().tolist()
                    values = df.loc[session, :].values.flatten().tolist()
                    fps = int(len(values) / 6)
                    signal.raw.append({'data': values, 'fps':fps})
                    print(signal.raw)

            else:
                raise ValueError(
                    "Invalid argument! %s signal is not present in dataset %s.\n" % (signal.name, self.name))
