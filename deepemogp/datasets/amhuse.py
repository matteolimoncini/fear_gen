import csv
import pandas as pd
from .dataset import Dataset
from scipy import io


class AMHUSE(Dataset):
    """
    Class to handle the AMHUSE dataset
    
    from http://amhuse.phuselab.di.unimi.it/
    
    Giuseppe Boccignone, Donatello Conte, Vittorio Cuculo, and Raffaella Lanzarotti. 
    2017. AMHUSE: a multimodal dataset for HUmour SEnsing. In Proceedings of the 19th ACM International Conference on Multimodal Interaction (ICMI 2017).
    ACM, New York, NY, USA, 438-445. DOI: https://doi.org/10.1145/3136755.3136806
    
    """

    # path of the directory containing physiological signals
    signal_folder = '/media/paolo/Volume/matteo/unimi/tesi_master/code/dataset/amhuse'
    # path of the directory containing emotional annotations
    anno_folder = '/home/vcuculo/Datasets/AMHUSE/sample_annotations/gs_annotations_1_to_36.mat'
    # path of the directory containing extracted fiducial points
    video_folder = '/home/vcuculo/Datasets/AMHUSE/fiducial_points/'

    def __init__(self, name='AMHUSE', signals=None, subjects=None, annotations=None):
        super(AMHUSE, self).__init__(name, signals, subjects, annotations)

        if signals is None:  # error!
            raise ValueError("Invalid argument! signals cannot be empty.\n")

        if subjects is None:  # error!
            raise ValueError("Invalid argument! subjects cannot be empty.\n")

        self.signals = signals  # contains the extracted signals data
        self.subjects = subjects  # contains the considered dataset subjects
        # structured as 'person_session'
        # (eg. 1_1 corresponds to subject 1, session 1)
        self.annotations = annotations  # contains the emotional annotations

        self.loadData()

        if annotations is not None:
            self.loadAnnotations()

    def loadData(self):

        ''' 
        Physiological data in AMHUSE dataset are structured in 
        different csv files, one per each subject. The csv file consists
        of 7 columns: timestamp, conductance (microSiemens), conductance (Volt),
        resistance (Ohm), temperature (Celsius deg), heart rate (bpm), oxysat (perc)
        recorded at 40Hz
        '''

        for signal in self.signals:  # loop over all considered signals

            if signal.name is 'EDA':

                for subject in self.subjects:  # loop over all considered subjects

                    person, session = subject.split("_")

                    print(">> Loading %s for subject %s and session %s from dataset %s" % (
                    signal.name, person, session, self.name))

                    # read raw data from CSV file
                    # csv_file = self.signal_folder + '/' + person + '/' + session + '/' + 'physio_' + session +'.csv'
                    csv_file = self.signal_folder + '/' + 'physio_1' + '.csv'
                    df = pd.read_csv(csv_file, sep=',')

                    # calculate original frame rate
                    fps = int(len(df.index) / (df['timestamp'].iloc[-1] / 10 ** 3))
                    # save raw data
                    signal.raw.append({'data': df['conductance_us'], 'fps': fps})


            elif signal.name is 'SKT':

                for subject in self.subjects:  # loop over all considered subjects

                    person, session = subject.split("_")

                    print(">> Loading %s for subject %s and session %s from dataset %s" % (
                    signal.name, person, session, self.name))

                    # read raw data from CSV file
                    # csv_file = self.signal_folder + '/' + person + '/' + session + '/' + 'physio_' + session +'.csv'
                    csv_file = self.signal_folder + '/' + 'physio_1' + '.csv'

                    df = pd.read_csv(csv_file, sep=',')

                    # calculate original frame rate
                    fps = int(len(df.index) / (df['timestamp'].iloc[-1] / 10 ** 3))
                    # save raw data
                    signal.raw.append({'data': df['temperature'], 'fps': fps})


            elif signal.name is 'HR':

                for subject in self.subjects:  # loop over all considered subjects

                    person, session = subject.split("_")

                    print(">> Loading %s for subject %s and session %s from dataset %s" % (
                    signal.name, person, session, self.name))

                    # read raw data from CSV file
                    # csv_file = self.signal_folder + '/' + person + '/' + session + '/' + 'physio_' + session +'.csv'
                    csv_file = self.signal_folder + '/' + 'physio_1' + '.csv'
                    df = pd.read_csv(csv_file, sep=',')

                    # calculate original frame rate
                    fps = int(len(df.index) / (df['timestamp'].iloc[-1] / 10 ** 3))
                    # save raw data
                    signal.raw.append({'data': df['hr'], 'fps': fps})


            elif signal.name is 'FP_OF':

                for subject in self.subjects:  # loop over all considered subjects

                    person, session = subject.split("_")

                    print(">> Loading %s for subject %s and session %s from dataset %s" % (
                    signal.name, person, session, self.name))

                    # read raw data from CSV file
                    csv_file = self.video_folder + '/' + subject + '.txt'
                    df = pd.read_csv(csv_file, sep=',', skipinitialspace=True,
                                     usecols=[x for x in [1] + list(range(10, 146))])

                    # calculate original frame rate 
                    fps = int(len(df.index) / df['timestamp'].iloc[-1])
                    # save raw data
                    signal.raw.append({'data': df, 'fps': fps})


            else:
                raise ValueError(
                    "Invalid argument! %s signal is not present in dataset %s.\n" % (signal.name, self.name))

    def loadAnnotations(self):
        '''
        AMHUSE dataset contains the annotations performed by 4 assistants (three males, one female)
        using the DANTE web-based annotation tool. Data are provided separately for each assistant,
        or combined in a mat file as gold standard calculated adoptin EWE (Evaluator Weighted Estimator)
        '''
        mat = io.loadmat(self.anno_folder)
        gs = mat['gs']

        for anno in self.annotations:
            for subject in self.subjects:  # loop over all considered subjects

                person, session = subject.split('_')
                print(">> Loading %s emotional annotations for subject %s and session %s from dataset %s" % (
                anno.name, person, session, self.name))

                l = gs[0, int(person) - 1][0, int(session) - 1][anno.name][0, 0]

                # save raw data
                anno.raw.append({'data': l, 'fps': 25})
