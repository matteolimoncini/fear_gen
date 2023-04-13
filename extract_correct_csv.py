# import
import os

import numpy as np
import pandas as pd

# Info: Subject codes 34 - 40 don't exist (switch of experimenter)
notvalid = [x for x in range(34, 41)]

# 3 pain rating decreased to non-painful levels (i.e. < 4) until the end of the generalization phase (11, 20, 42)
notvalid.extend([11, 20, 42])

# 1 experiment aborted due to insufficient recognition performance of CS+ and CS- within the differentiation task (#9)
notvalid.extend([9])

# 1 too many extrasystoles to interpolate (#12)
notvalid.extend([12])

# 1 eye tracker calibration not successful (#25)
notvalid.extend([25])

# 2 too many missing values (> 20%) during generalization phase (29, 30)
notvalid.extend([29, 30])

# 10 non-responsive to UCS (8, 10, 12, 14, 18, 24, 29, 49, 53, 55)
notvalid.extend([8, 10, 14, 18, 24, 49, 53, 55])

# why we have inserted this?
notvalid.extend([3])

# 19 due to error in processing eda. division by zero with new method
notvalid.extend([19])

# (49, 11, 20, 42, 29, 30, 25, 12, 8, 10, 14, 18, 24, 53, 55)
valid_pupil_eda = [ele for ele in range(1, 56) if ele not in notvalid]

# variabili globali
NUM_TRIALS = 160
LATENCY_HR = 250
LATENCY_EDA = 5000
LATENCY_PUPIL = 1000
SAMPLING_RATE_HR = 500
SAMPLING_RATE_EDA = 500
SAMPLING_RATE_PUPIL = 100

def extract_only_valid_subject():
    valid_final = []
    for i in valid_pupil_eda:
        df = pd.read_csv('./data/sync_signals_raw/eda_csv/' + str(
            i) + '_eda.csv')
        if df.shape[0] == 160:
            valid_final.append(i)
        else:
            print(f'subject: {i}, num trials: {df.shape[0]}')
    return valid_final


def read_correct_subject_csv(subject):
    if int(subject) < 10:
        return '0'+str(subject)
    else:
        return str(subject)

# extract subjects with different sias and lds score
def anxious_subjects(path, n, type_='top'):
    #os.chdir('..')
    valid_subjects = extract_only_valid_subject()
    #os.chdir('pyro')
    df = pd.read_csv(path).dropna().reset_index(drop=True)
    df = df[df.subject.isin(valid_subjects)]
    df['subject'] = [int(x) for x in df['subject']]
    if type_=='top':
        return df.sort_values(by=df.columns[1], ascending=False).subject[:n].values
    else:
        return df.sort_values(by=df.columns[1], ascending=False).subject[-n:].values

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

    print(len(extract_only_valid_subject()))

