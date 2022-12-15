# import

import csv
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from scipy import stats

#
notvalid = [x for x in range(34,41)]
notvalid.extend([9,11,20,25,42])
valid_patients_pupil = [ele for ele in range(1,56) if ele not in notvalid]

notvalid = [x for x in range(34,41)]
notvalid.append(9)
valid_patients_eda = [ele for ele in range(1,56) if ele not in notvalid]

valid_pupil_eda = list(set(valid_patients_eda).intersection(set(valid_patients_pupil)))

notvalid = [x for x in range(34, 41)]
notvalid= notvalid + [9,11,12,20]
valid_patients_hr = [ele for ele in range(1, 56) if ele not in notvalid]

# variabili globali
NUM_TRIALS = 160
LATENCY_HR = 250
LATENCY_EDA = 5000
LATENCY_PUPIL = 1000
SAMPLING_RATE_HR = 500
SAMPLING_RATE_EDA = 500
SAMPLING_RATE_PUPIL = 100

# physio signals functions

def create_df_one_subj(df:pd.DataFrame,indexes_trial:list,type_data:str,sampling_rate=500,latency=0,len_trial=6) -> pd.DataFrame:
    """
    Modify pd dataframes trasforming datas into #numtrials rows and 3k columns similar to pupil data
    This function also fix latency

    :param df: df with a row for each value
    :param indexes_trial: list that cointains indexes of the df where starts trials
    :param type_data: str that indicates 'eda' or 'hr'
    :param sampling_rate: in hz default 500
    :param latency: in seconds default 0
    :param len_trial: in seconds default 6
    :return: DataFrame with 160 rows (1 per trial and 3k columns)
    """
    num_data = (len_trial*sampling_rate)-1
    names_col = np.arange(0,num_data+1).astype(str)
    df_all_trials = pd.DataFrame()
    for i,index in enumerate(indexes_trial):
        start_lat= index + (latency/1000*sampling_rate)
        end_lat = start_lat + num_data
        df_single_trial = df.loc[start_lat:end_lat][[type_data]]
        min_len= len(df_single_trial.index)
        df_single_trial.columns = [str(i+1)]
        df_trasp = df_single_trial.T
        df_trasp.columns =names_col[:min_len]
        df_all_trials = pd.concat([df_all_trials, df_trasp], axis=0)
    return df_all_trials


def from_csv_to_df (subject_number,valid_patients,type_data,path="tmp_eda") -> pd.DataFrame:
    """
    Read csv files from @param path and return a simple dataframe with eda (or hr) and trigger data

    :param subject_number:
    :param valid_patients:
    :param type_data:
    :param path:
    :return:
    """
    df = dict()
    if type_data not in ['eda','hr']:
        print("ERROR TYPE_DATA PARAM (only eda or hr)")
        return pd.DataFrame(df)

    if subject_number not in valid_patients:
        print('subject number not valid, probably this patient has not valid '+type_data+' signal')
        return pd.DataFrame(df)

    if subject_number < 10:
        subject_number = '0' + str(subject_number)
    path_csv = str(path+"/tmp_eda"+ str(subject_number) + ".csv")
    pat_ = pd.read_csv(path_csv)
    if type_data == 'eda':
        df = {'subject': subject_number, 'eda': pat_['CH1'], 'trigger': pat_['CH28']}
    if type_data == 'hr':
        df = {'subject': subject_number, 'hr': pat_['CH2'], 'trigger': pat_['CH28']}

    return pd.DataFrame(df)


def indexes_trial_start (df:pd.DataFrame)-> list:
    """
    Calculate indexes where trials starts
    :param df: a pandas DataFrame with rectangular function with value = 0 or value = 5 in column 2
    :return: a list that contains indexes of firsts value with value = 5
    """
    start_trials = [] #TODO optimize this function.too much time
    prec = -1
    for i in df.iterrows():
        actual = i[1][2]
        if actual == 5 and prec == 0:
            start_trials.extend([i[0]])
        prec = actual
    return start_trials

def create_csv_signals_physio (subj_num, valid_patients,type_signal) -> None:
    """
    Function to create csv files into eda_csv or hr_csv folder
    :param subj_num:
    :param valid_patients:
    :param type_signal:
    :return:
    """
    if type_signal not in ['eda','hr']:
        print("ERROR TYPE_DATA PARAM (only eda or hr)")
        return
    df = from_csv_to_df(subj_num,valid_patients,type_data=type_signal)
    indexes_trial_ = indexes_trial_start(df)
    #print('len ind trial '+str(len(indexes_trial_)))
    if type_signal== 'eda':
        latency = LATENCY_EDA
    if type_signal== 'hr':
        latency = LATENCY_HR
    df_eda_subj = create_df_one_subj(df,indexes_trial=indexes_trial_,latency=latency,type_data=type_signal)
    name=type_signal+'_csv/'+str(subj_num)+'_'+type_signal+'.csv'
    df_eda_subj.to_csv(name,index=False)

def from_csv_to_df_pupil(subject:int, latency:int, sampling_rate: int) -> pd.DataFrame:
    """
    Read in current folder one tmp csv well formatted with pupil data
    :param subject:
    :param latency: msec
    :param sampling_rate: hz
    :return: Dataframe with pupil data of the subject
    """
    name = 'tmp_pupil/tmp_pupil'+str(subject)+'.csv'
    df_pupil = pd.read_csv(name, sep=',')
    delay_before_onset_stim = 1000
    len_col = int((latency+delay_before_onset_stim) / 1000 * sampling_rate)
    name_col = 'pd'
    columns = ['trial']
    for i in range(1, len_col+1):
        columns.append(name_col+str(i))
    df_pupil = df_pupil.drop(columns, axis=1)
    columns = np.arange(0, len(df_pupil.columns))
    df_pupil.columns = columns
    return df_pupil


if __name__ == '__main__':
    print()
