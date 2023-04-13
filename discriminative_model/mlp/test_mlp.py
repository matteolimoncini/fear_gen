import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import extract_correct_csv
from tqdm import tqdm

valid_subject = extract_correct_csv.extract_only_valid_subject()
valid_subject.remove(50)
valid_subject.remove(51)

scaler = StandardScaler()
# test tuning MLP
first_layer_neurons = np.arange(10, 150, 10)
second_layer_neurons = np.arange(10, 150, 10)


best_hyperP = pd.DataFrame(columns=['first layer', 'second layer', 'train', 'test'])

for first in first_layer_neurons:
    for second in second_layer_neurons:
        mean_test = []
        mean_train = []
        for sub in tqdm(valid_subject):
            string_sub = extract_correct_csv.read_correct_subject_csv(sub)
            df_ = pd.read_csv('data/LookAtMe_0'+string_sub+'.csv', sep='\t')
            y = np.array(list([int (d > 2) for d in df_['rating']]))
            y = y[48:]
            X1 = pd.read_csv('data/features_4_2/hr/'+str(sub)+'.csv')
            X1 = pd.DataFrame(scaler.fit_transform(X1))
            X2 = pd.read_csv('data/features_4_2/eda/'+str(sub)+'.csv')
            X2 = pd.DataFrame(scaler.fit_transform(X2))
            X3 = pd.read_csv('data/features_4_2/pupil/'+str(sub)+'.csv')
            X3 = pd.DataFrame(scaler.fit_transform(X3))
            X = pd.concat([X1, X2, X3], axis=1)
            X = X[48:]
            X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123, stratify=y)
            clf = MLPClassifier(hidden_layer_sizes=(first, second),
                                max_iter=3000,
                                learning_rate='adaptive',
                                random_state=123)\
                .fit(X_train, y_train)
            mean_train.append(clf.score(X_train, y_train))
            mean_test.append(clf.score(X_test, y_test))

        row_dict = {'first layer': first,
                    'second layer': second,
                    'train': np.array(mean_train).mean(),
                    'test': np.array(mean_test).mean()}
        best_hyperP = pd.concat([best_hyperP, pd.DataFrame(data=row_dict, index=np.arange(1))], ignore_index=True)
best_hyperP.to_csv('output_mlp_multi_2layers.csv')
