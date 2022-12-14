import csv
import pandas as pd
import numpy as np
from .dataset import Dataset
from scipy import io


class RECOLA(Dataset):
    """
    Class to handle the RECOLA dataset
    
    from https://diuf.unifr.ch/diva/recola/
    
    F. Ringeval, A. Sonderegger, J. Sauer and D. Lalanne,
    "Introducing the RECOLA Multimodal Corpus of Remote Collaborative and Affective Interactions",
    2nd International Workshop on Emotion Representation, Analysis and Synthesis in Continuous Time and Space (EmoSPACE),
    in Proc. of IEEE Face & Gestures 2013, Shanghai (China), April 22-26 2013.
    
    """

    # path of the directory containing physiological signals
    signal_folder = '/home/vcuculo/Datasets/RECOLA/RECOLA-Biosignals-recordings/'
    # path of the directory containing emotional annotations
    anno_folder = '/home/vcuculo/Datasets/RECOLA/RECOLA-Annotation/emotional_behaviour/'
    # path of the directory containing video signals
    video_folder = '/home/vcuculo/Datasets/RECOLA/RECOLA-Video-features/'

    def __init__(self, name='RECOLA', signals=None, subjects=None, annotations=None):
        super(RECOLA, self).__init__(name, signals, subjects, annotations)

        if signals is None:  # error!
            raise ValueError("Invalid argument! signals cannot be empty.\n")

        if subjects is None:  # error!
            raise ValueError("Invalid argument! subjects cannot be empty.\n")

        self.signals = signals  # contains the extracted signals data
        self.subjects = subjects  # contains the considered dataset subjects
        self.annotations = annotations  # contains the emotional annotations

        self.loadData()

        if annotations is not None:
            self.loadAnnotations()

    def loadData(self):

        ''' 
        Physiological data in RECOLA dataset are structured in 
        different csv files, one per each subject. The csv file consists
        of 3 columns: time, EDA, ECG, recorded at 1KHz
        '''

        for signal in self.signals:  # loop over all considered signals

            if signal.name is 'EDA':

                for subject in self.subjects:  # loop over all considered subjects

                    print(">> Loading %s for subject %s from dataset %s" % (signal.name, subject, self.name))

                    # read raw data from CSV file
                    csv_file = self.signal_folder + subject + '.csv'
                    df = pd.read_csv(csv_file, sep=';')
                    # calculate original frame rate
                    fps = int(len(df.index) / df['time'].iloc[-1])
                    # save raw data
                    signal.raw.append({'data': df['EDA'], 'fps': fps})


            elif signal.name is 'ECG':

                for subject in self.subjects:  # loop over all considered subjects

                    print(">> Loading %s for subject %s from dataset %s" % (signal.name, subject, self.name))

                    # read raw data from CSV file
                    csv_file = self.signal_folder + subject + '.csv'
                    df = pd.read_csv(csv_file, sep=';', skipinitialspace=True)
                    # calculate original frame rate
                    fps = int(len(df.index) / df['time'].iloc[-1])
                    # save raw data
                    signal.raw.append({'data': df['ECG '], 'fps': fps})


            elif signal.name is 'FP_OF':

                for subject in self.subjects:  # loop over all considered subjects

                    print(">> Loading %s for subject %s from dataset %s" % (signal.name, subject, self.name))

                    # read raw data from CSV file
                    csv_file = self.video_folder + 'extra/FP/' + subject + '.csv'
                    df = pd.read_csv(csv_file, sep=',', skipinitialspace=True,
                                     usecols=[x for x in [1] + list(range(4, 140))])

                    # calculate original frame rate
                    fps = len(df.index) / df['timestamp'].iloc[-1]
                    # save raw data
                    signal.raw.append({'data': df, 'fps': fps})


            # elif signal.name is 'AU':

            #     # specify the arff fields to extract
            #     fields = ['time_code', 'AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', \
            #     'AU11', 'AU12', 'AU15', 'AU17', 'AU20', 'AU23', 'AU24', 'AU25']

            #     for subject in self.subjects: # loop over all considered subjects

            #         print ">> Loading %s for subject %s from dataset %s" % (signal.name, subject, self.name)

            #         # read raw data from mat files
            #         data = io.arff.loadarff(self.video_folder + subject + '.arff')
            #         df = pd.DataFrame(data[0][fields])

            #         # calculate original frame rate
            #         fps = int(len(df.index)/df['time_code'].iloc[-1])
            #         # save raw data
            #         signal.raw.append({'data': df.iloc[:,1:], 'fps': fps})

            # elif signal.name is 'AU_OF':

            #     # specify the mat fields to extract
            #     fields = ['timestamp', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', \
            #     'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', \
            #     'AU26_r', 'AU45_r']

            #     for subject in self.subjects: # loop over all considered subjects

            #         print ">> Loading %s for subject %s from dataset %s" % (signal.name, subject, self.name)

            #         # read raw data from mat files
            #         mat = io.loadmat(self.video_folder + 'extra/AU' + subject + '_au_openface.mat')
            #         all_data = mat['data']
            #         df = pd.DataFrame(np.hstack(all_data[fields][0,0]))

            #         # calculate original frame rate
            #         fps = int(len(df.index)/df[0].iloc[-1])
            #         # save raw data
            #         signal.raw.append({'data': df.iloc[:,1:], 'fps': fps})

            else:
                raise ValueError(
                    "Invalid argument! %s signal is not present in dataset %s.\n" % (signal.name, self.name))

    def loadAnnotations(self):
        '''
        RECOLA dataset contains the annotations performed by the six assistants (three males, three females)
        using the ANNEMO web-based annotation tool. Data are provided separately for each participant and each assistant,
        with a framerate of 40ms for the affective behaviours (arousal and valence).
        '''

        for anno in self.annotations:
            for subject in self.subjects:  # loop over all considered subjects

                print(">> Loading %s emotional annotations for subject %s from dataset %s" % (
                anno.name, subject, self.name))

                # read raw data from CSV file
                csv_file = self.anno_folder + anno.name + '/' + subject + '.csv'
                df = pd.read_csv(csv_file, sep=';')
                # calculate original frame rate
                fps = len(df.index) / df['time'].iloc[-1]
                # save raw data
                anno.raw.append({'data': df.iloc[:, 1:], 'fps': fps})
