import numpy as np
import pandas as pd
from .dataset import Dataset
from scipy import io


class FEAR(Dataset):
    # class used to use this package with the fear generalization dataset
    # path of the directory containing physiological signals
    signal_folder = 'data/sync_signals_raw'

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
        list_less_trials = [13, 15, 16, 17, 23, 26, 27, 28, 31, 32, 33]
        bool_eda_null = False
        for signal in self.signals:  # loop over all considered signals

            if signal.name == 'EDA':

                for subject in self.subjects:  # loop over all considered subjects

                    if '_' in subject:
                        person, session = subject.split("_")
                        print(">> Loading %s for subject %s and session %s from dataset %s" % (
                            signal.name, person, session, self.name))
                        session = int(session) - 1
                        csv_file = self.signal_folder + '/eda_csv/' + person + '_eda.csv'
                        df = pd.read_csv(csv_file)
                        values = df.loc[session, :].values.flatten().tolist()
                        fps = int(len(values) / 6)
                        signal.raw.append({'data': values, 'fps': fps})

                    else:
                        person = subject
                        print(">> Loading %s for subject %s and all sessions from dataset %s" % (
                            signal.name, person, self.name))
                        csv_file = self.signal_folder + '/eda_csv/' + person + '_eda.csv'
                        df = pd.read_csv(csv_file)

                        if int(person) in list_less_trials:
                            print("EDA: The last row of the DataFrame contains null values.")
                            df = df[:-1]

                        for i in df.index:
                            values = df.loc[i, :].values.flatten().tolist()
                            fps = int(len(values) / 6)
                            signal.raw.append({'data': values, 'fps': fps})
                    # print(signal.raw[:10])

            elif signal.name == 'PUPIL':

                for subject in self.subjects:  # loop over all considered subjects

                    if '_' in subject:
                        person, session = subject.split("_")
                        print(">> Loading %s for subject %s and session %s from dataset %s" % (
                            signal.name, person, session, self.name))
                        session = int(session) - 1
                        csv_file = self.signal_folder + '/pupil_csv/' + person + '_pupil.csv'
                        df = pd.read_csv(csv_file)
                        values = df.loc[session, :].values.flatten().tolist()
                        fps = int(len(values) / 5)
                        signal.raw.append({'data': values, 'fps': fps})
                    else:
                        person = subject
                        print(">> Loading %s for subject %s and all sessions from dataset %s" % (
                            signal.name, person, self.name))
                        csv_file = self.signal_folder + '/pupil_csv/' + person + '_pupil.csv'
                        df = pd.read_csv(csv_file)

                        if int(person) in list_less_trials:
                            print("PUPIL: The last row of the DataFrame contains null values.")
                            df = df[:-1]

                        for i in df.index:
                            values = df.loc[i, :].values.flatten().tolist()
                            fps = int(len(values) / 5)
                            signal.raw.append({'data': values, 'fps': fps})
                    #print(signal.raw[:10])

            elif signal.name == 'ECG':

                for subject in self.subjects:  # loop over all considered subjects

                    if '_' in subject:
                        person, session = subject.split("_")
                        print(">> Loading %s for subject %s and session %s from dataset %s" % (
                            signal.name, person, session, self.name))
                        session = int(session) - 1
                        csv_file = self.signal_folder + '/hr_csv/' + person + '_hr.csv'
                        df = pd.read_csv(csv_file)
                        values = df.loc[session, :].values.flatten().tolist()
                        fps = int(len(values) / 6)
                        signal.raw.append({'data': values, 'fps': fps})

                    else:
                        person = subject
                        print(">> Loading %s for subject %s and all sessions from dataset %s" % (
                            signal.name, person, self.name))
                        csv_file = self.signal_folder + '/hr_csv/' + person + '_hr.csv'
                        df = pd.read_csv(csv_file)

                        if int(person) in list_less_trials:
                            print("ECG: The last row of the DataFrame contains null values.")
                            df = df[:-1]

                        for i in df.index:
                            values = df.loc[i, :].values.flatten().tolist()
                            fps = int(len(values) / 6)
                            signal.raw.append({'data': values, 'fps': fps})
                    #print(signal.raw[:10])

            else:
                raise ValueError(
                    "Invalid argument! %s signal is not present in dataset %s.\n" % (signal.name, self.name))
