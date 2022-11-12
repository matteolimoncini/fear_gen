import csv
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyreadr import pyreadr

# setup plots
plt.rcParams['figure.figsize'] = [15, 5]  # Bigger images
plt.rcParams['font.size'] = 16

# remove patients non valid for eda
notvalid = [x for x in range(34, 41)]
notvalid.append(9)
valid_patients_eda = [ele for ele in range(1, 56) if ele not in notvalid]

# remove patients non valid for pupil
notvalid = [x for x in range(34, 41)]
notvalid.extend([9, 20, 25, 42])
valid_patients_pupil = [ele for ele in range(1, 56) if ele not in notvalid]

# select patients with either pupil and eda data
valid_pupil_eda = list(set(valid_patients_eda).intersection(set(valid_patients_pupil)))


def extract_pupil_by_subject(subject_number: int) -> list:
    if subject_number not in valid_patients_pupil:
        print('subject number not valid, probably this patient has not valid pupil signals')
        return []
    if subject_number < 10:
        subject_number = '0' + str(subject_number)
    pupil1 = pd.read_csv('../osfstorage-archive/eye/pupil/Look0' + subject_number + '_pupil.csv', sep=';')
    for i in pupil1.columns:
        if i != 'trial':
            for j in pupil1.index:
                pupil1.loc[j, i] = pupil1.loc[j, i].replace(',', '.')
    cols = pupil1.columns.drop('trial')

    pupil1[cols] = pupil1[cols].apply(pd.to_numeric, errors='coerce')

    # convert all datas into one list
    pat1_pupil = []
    for i in range(160):
        colonne = pupil1.columns.drop(['trial'])
        for colonna in colonne:
            pat1_pupil.append(pupil1.loc[i][colonna])
    pat1_pupil

    return pat1_pupil


def extract_eda_by_subject(subject_number: int) -> list:
    if subject_number not in valid_patients_eda:
        print('subject number not valid, probably this patient has not valid eda signals')
        return []
    if subject_number < 10:
        subject_number = '0' + str(subject_number)
    pat1_eda = pd.read_csv("../tmp_eda" + subject_number + ".csv")['CH1']
    pat1_eda.to_numpy()


if __name__ == '__main__':
    print(extract_pupil_by_subject(1))
