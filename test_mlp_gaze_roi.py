import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import extract_correct_csv
from tqdm import tqdm
import re
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def custom_parse_data(X):
    res_x = []
    for i in range(len(X)):
        string = X[i]
        new_string = re.sub(r'\s+', ',', string)
        string_list = list(new_string)
        string_list[1] = ''
        new_string = ''.join(string_list)
        res_x.append(np.fromstring(new_string.strip('[]'), dtype=float, sep=','))

    return np.array(res_x)


valid_subject = extract_correct_csv.extract_only_valid_subject()
valid_subject.remove(50)
valid_subject.remove(51)

scaler = StandardScaler()
# test tuning MLP
first_layer_neurons = np.arange(10, 300, 10)
second_layer_neurons = np.arange(10, 300, 10)

best_hyperP = pd.DataFrame(columns=['first layer', 'second layer', 'train', 'test'])

df_ = pd.read_csv('data/gaze/joined_fixation.csv')
# remove index column
df_ = df_.drop(columns=df_.columns[0])
# replace NaN with 'other'
df_['ROI'] = df_['ROI'].replace(np.NaN, 'other')
# parse fixation feature column to convert this in value
feat = custom_parse_data(df_['Fixation feature'])
df_['Fixation feature'] = [x for x in feat]
# mean features with same ROI
result = df_.groupby(['Subject', 'Trial', 'ROI'], as_index=False)['Fixation feature'].mean()
# insert an array of zeros if one subject in one trial has no fixations on that ROI
result_modified = result.copy(deep=True)
for sub in result['Subject'].unique():
    df = result[result['Subject'] == sub]
    for trial in df['Trial'].unique():
        if 'mouth_nose' not in list(df[df['Trial'] == trial]['ROI']):
            result_modified = result_modified.append(
                {'Subject': sub, 'Trial': trial, 'ROI': 'mouth_nose', 'Fixation feature': [0] * 13}, ignore_index=True)
        if 'eye' not in list(df[df['Trial'] == trial]['ROI']):
            result_modified = result_modified.append(
                {'Subject': sub, 'Trial': trial, 'ROI': 'eye', 'Fixation feature': [0] * 13}, ignore_index=True)
        if 'other' not in list(df[df['Trial'] == trial]['ROI']):
            result_modified = result_modified.append(
                {'Subject': sub, 'Trial': trial, 'ROI': 'other', 'Fixation feature': [0] * 13}, ignore_index=True)

for first in first_layer_neurons:
    for second in second_layer_neurons:
        mean_test_type = []
        mean_train_type = []
        for sub in tqdm(valid_subject):
            string_sub = extract_correct_csv.read_correct_subject_csv(sub)
            # data of one subject
            df_sub = result_modified[result_modified['Subject'] == sub]
            # convert rating into 0/1
            df_look = pd.read_csv('./data/LookAtMe_old/LookAtMe_0' + string_sub + '.csv', sep='\t')
            y = np.array(list([int(d > 2) for d in df_look['rating']]))
            y = y[48:]
            # normalize feature and convert from [13][13][13] into [39]
            X1 = df_sub[df_sub['ROI'] == 'eye']
            X1_norm = pd.DataFrame(scaler.fit_transform(list(X1['Fixation feature'])))
            X2 = df_sub[df_sub['ROI'] == 'mouth_nose']
            X2_norm = pd.DataFrame(scaler.fit_transform(list(X2['Fixation feature'])))
            X3 = df_sub[df_sub['ROI'] == 'other']
            X3_norm = pd.DataFrame(scaler.fit_transform(list(X3['Fixation feature'])))
            X_norm = pd.concat([X1_norm, X2_norm, X3_norm], axis=1)
            # remove first 48 learning trials
            X_norm = X_norm[48:]
            X_norm = pd.DataFrame(X_norm)

            # add ROI
            lookatme = pd.read_csv('data/newLookAtMe/newLookAtMe' + string_sub + '.csv')
            significant_ROI = pd.read_csv('data/newLookAtMeROI/LookAtMe' + string_sub + '.csv')
            significant_ROI = significant_ROI['ROI_change']
            significant_ROI = [1 if x == 'mouth_nose' else 0 for x in significant_ROI]
            significant_ROI = pd.DataFrame(significant_ROI)
            X_norm = pd.concat([X_norm, significant_ROI], axis=1)

            X_norm = pd.DataFrame(X_norm)

            X_norm = X_norm.reset_index().drop(columns=('index'))

            X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=123, stratify=y)

            clf = MLPClassifier(hidden_layer_sizes=(first, second), max_iter=3000, learning_rate='adaptive',
                                random_state=123).fit(X_train, y_train)
            mean_train_type.append(clf.score(X_train, y_train))
            mean_test_type.append(clf.score(X_test, y_test))

        row_dict = {'first layer': first,
                    'second layer': second,
                    'train': np.array(mean_train_type).mean(),
                    'test': np.array(mean_test_type).mean()}
        best_hyperP = pd.concat([best_hyperP, pd.DataFrame(data=row_dict, index=np.arange(1))], ignore_index=True)
best_hyperP.to_csv('./output/output_mlp_gaze_roi.csv')
