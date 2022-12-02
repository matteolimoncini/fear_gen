import csv
import os.path

import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyreadr import pyreadr
import matplotlib
from tqdm import tqdm
from scipy import stats


NUM_TRIALS = 160
LATENCY_HR = 250
LATENCY_EDA = 5000
LATENCY_PUPIL = 1000
SAMPLING_RATE_HR = 500
SAMPLING_RATE_EDA = 500
SAMPLING_RATE_PUPIL = 100


def create_single_signal_lists(df_pupil, df_eda, df_hr):
    """
    Create 3 list, one for pupil, one for eda, one for ecg with all signal with df

    :param df_pupil: df with all pupil data. one row for trial. #columns = 1/sampling rate * len trial. each value contains pupil dilation
    :param df_eda: df with all eda data. one row for trial. #columns = 1/sampling rate * len trial. each value contains eda data
    :param df_hr: df with all hr data. one row for trial. #columns = 1/sampling rate * len trial. each value contains hr data. hr converted in hb

    :return: 3 lists
    """

    list_pupil, list_eda, list_hr = [], [], []
    for i in range(len(df_pupil)):
        raw_list = df_pupil.loc[i, :].values.flatten().tolist()
        list_pupil = list_pupil + raw_list
        list_pupil.extend([float('NaN') for x in range(100)])
    for i in range(len(df_eda)):
        raw_list = df_eda.loc[i, :].values.flatten().tolist()
        list_eda = list_eda + raw_list
    for i in range(len(df_hr)):
        raw_list = df_hr.loc[i, :].values.flatten().tolist()
        df, info = nk.ecg_process(raw_list, sampling_rate=SAMPLING_RATE_HR)
        ecg_rate_trial = df['ECG_Rate']
        list_hr = list_hr + list(ecg_rate_trial)
    return list_pupil, list_eda, list_hr


def get_data_from_subject(subject_number: int, list_type_data = list):

    eda_subj = pd.read_csv('eda_csv/' + str(subject_number) + '_eda.csv')
    hr_subj = pd.read_csv('hr_csv/' + str(subject_number) + '_hr.csv')
    pupil_subj = pd.read_csv('pupil_csv/' + str(subject_number) + '_pupil.csv')

    list_pupil, list_eda, list_hr = create_single_signal_lists(pupil_subj, eda_subj, hr_subj)

    return list_pupil, list_eda, list_hr


def export_data_to_npy(filename: str, sliding_win:int):
    """

    :param path:
    :param filename:
    :param data_as_array:
    :return:
    """
    hr = np.array(extract_feature_basic(sliding_win, 'hr', 2, 500))
    eda = np.array(extract_feature_basic(sliding_win, 'eda', 2, 500))
    pupil = np.array(extract_feature_basic(sliding_win, 'pupil', 2, 100))
    d1 = {'hr':hr, 'eda':eda, 'pupil':pupil}
    np.save(filename, d1)


def read_csv(subject_number:int, type:str):
    csv_ = ''
    if type == 'eda':
        csv_ = pd.read_csv('eda_csv/' + str(subject_number) + '_eda.csv')
    elif type == 'hr':
        csv_ = pd.read_csv('hr_csv/' + str(subject_number) + '_hr.csv')
    elif type == 'pupil':
        csv_ = pd.read_csv('pupil_csv/' + str(subject_number) + '_pupil.csv')

    return csv_


def extract_feature_basic(sliding_window, type_: str, subject: int, frequency: int):
    csv_ = read_csv(subject, type_)
    generic = []
    start = 0
    for i in range(1, int(csv_.shape[1]/(frequency*sliding_window))+1):
        print(i)
        end = int(start+frequency*sliding_window)
        new_df = csv_.iloc[:, start:end]
        mean = new_df.mean(axis=1)
        if i == 1:
            generic = mean
        else:
            generic = np.column_stack((generic, mean))
        start = end + 1
    return generic


if __name__ == '__main__':
    path = 'prova.npy'
    if not os.path.exists(path):
        export_data_to_npy('prova.npy', 1)
    data = np.load('prova.npy', allow_pickle=True).item()
    print(data.keys())
