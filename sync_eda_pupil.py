import csv
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyreadr import pyreadr

#setup plots
plt.rcParams['figure.figsize'] = [15, 5]  # Bigger images
plt.rcParams['font.size']= 16


#remove patients non valid for eda
notvalid = [x for x in range(34,41)]
notvalid.append(9)
valid_patients_eda = [ele for ele in range(1,56) if ele not in notvalid]

#remove patients non valid for pupil
notvalid = [x for x in range(34,41)]
notvalid.extend([9,11,20,25,42])
valid_patients_pupil = [ele for ele in range(1,56) if ele not in notvalid]

#select patients with either pupil and eda data
valid_pupil_eda = list(set(valid_patients_eda).intersection(set(valid_patients_pupil)))


def read_csv_pupil_raw(subject_number:int) -> pd.DataFrame:
    if subject_number not in valid_patients_pupil:
        print('subject number not valid, probably this patient has not valid pupil signals')
        return pd.DataFrame()
    if subject_number < 10:
        subject_number = '0'+str(subject_number)
    pupil1 = pd.read_csv('../osfstorage-archive/eye/pupil/Look0'+str(subject_number)+'_pupil.csv', sep=';')
    for i in pupil1.columns:
        if i != 'trial':
            for j in pupil1.index:
                pupil1.loc[j, i] = pupil1.loc[j, i].replace(',', '.')
    cols = pupil1.columns.drop('trial')

    pupil1[cols] = pupil1[cols].apply(pd.to_numeric, errors='coerce')
    return pupil1


def create_csv_pupil():
    for subject in valid_patients_pupil:
        print(subject)
        pupil_i = read_csv_pupil_raw(subject)
        name = 'tmp_pupil' + str(subject) + '.csv'
        pupil_i.to_csv(name, index=False)

def read_csv_pupil(subject:int) -> pd.DataFrame:
    name = 'tmp_pupil'+str(subject)+'.csv'
    return pd.read_csv(name, sep=',')

def extract_pupil_by_subject(subject_number:int) -> list:
    pupil = read_csv_pupil_raw(subject_number)

    # convert all datas into one list
    pat1_pupil = []
    for i in range(160):
        colonne = pupil.columns.drop(['trial'])
        for colonna in colonne:
            pat1_pupil.append(pupil.loc[i][colonna])
    #pat1_pupil

    return pat1_pupil

def extract_eda_by_subject(subject_number:int) -> list:
    if(subject_number not in valid_patients_eda):
        print('subject number not valid, probably this patient has not valid eda signals')
        return []
    if(subject_number<10):
        subject_number= '0'+str(subject_number)
    path_csv = str("tmp_eda" + str(subject_number) + ".csv")
    pat_eda = pd.read_csv(path_csv)['CH1']
    return pat_eda.to_numpy()


def extract_maxpupil_trial(pupil_csv: pd.DataFrame) -> list:
    max_list = []
    for i in range(160):
        max_trial_list = list(pupil_csv.loc[i])[1:]
        max_ = max(max_trial_list)
        max_index = max_trial_list.index(max_)
        for j in range(700):
            if j == max_index:
                max_list.append(1)
            else:
                max_list.append(0)
    return max_list


def all_subject_pupil() -> pd.DataFrame:
    generic_df = pd.DataFrame(columns=['pupilDiameter', 'maxIndex', 'subject'])
    for i in valid_patients_pupil:
        subject = i
        #print(subject)
        person_i = read_csv_pupil_raw(subject)
        person_i_all_pupil = extract_pupil_by_subject(subject)
        max_list_i = extract_maxpupil_trial(person_i)
        dict_ = {'pupilDiameter': person_i_all_pupil, 'maxIndex': max_list_i, 'subject': [i for x in range(len(max_list_i))]}
        df_ = pd.DataFrame(dict_)
        generic_df = pd.concat([generic_df, df_], axis=0)
    time_ = np.arange(0, len(generic_df)/100, 0.01)
    generic_df['time'] = time_
    return generic_df


def resample_eda(eda_signal) -> list:
    eda_new = []
    for x in range(len(eda_signal)):
        if x % 5 == 0:
            eda_new.append(eda_signal[x])
    return eda_new

def all_subject_eda() -> pd.DataFrame:
    generic_df = pd.DataFrame(columns=['subject','phasic','phasic_peak'])
    for i in valid_patients_eda:
        eda = extract_eda_by_subject(i)

        eda = resample_eda(eda)

        signals, info = nk.eda_process(eda, sampling_rate=100, method="neurokit")
        df = {'subject':i,'phasic':signals['EDA_Phasic'],'phasic_peak':signals['SCR_Peaks']}
        df_=pd.DataFrame(df)
        generic_df = pd.concat([generic_df, df_], axis=0)
    generic_df['time'] = np.arange(0,len(generic_df)/100,0.01)
    return generic_df

if __name__ == '__main__':
    print(read_csv_pupil(55))




