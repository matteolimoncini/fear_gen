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

types_ = ['hr', 'pupil', 'eda']

best_hyperP = pd.DataFrame(columns=['type', 'first layer', 'second layer', 'train', 'test'])

for first in first_layer_neurons:
    for second in second_layer_neurons:
        mean_test_hr = []
        mean_train_hr = []
        for sub in tqdm(valid_subject):
            string_sub = extract_correct_csv.read_correct_subject_csv(sub)
            df_ = pd.read_csv('data/LookAtMe_0'+string_sub+'.csv', sep='\t')
            y = np.array(list([int(d > 2) for d in df_['rating']]))
            y = y[48:]

            for type_ in types_:
                X = pd.read_csv('data/features_4_2/' + type_ + '/' + str(sub) + '.csv')
                X = pd.DataFrame(scaler.fit_transform(X))
                X = X[48:]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
                clf = MLPClassifier(hidden_layer_sizes=(first, second), max_iter=3000, learning_rate='adaptive',
                                    random_state=123).fit(X_train, y_train)
                row_ = {'Subject': sub,
                        'Feature': type_,
                        'Train': clf.score(X_train, y_train),
                        'Test': clf.score(X_test, y_test)}
                best_hyperP = pd.concat([best_hyperP, pd.DataFrame(data=row_, index=np.arange(1))], ignore_index=True)
best_hyperP.to_csv('output_mlp_multi_2layers.csv')
