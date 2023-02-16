import numpy as np
import pandas as pd
import glob


def extract_conditioned_stimuli_rating(df):
    df_ = df[(df['morphing level'] == 1) | (df['morphing level'] == 6)]
    return df_['rating'].values


def extract_generalization_stimuli(df):
    df_ = df[(df['morphing level'] != 1) & (df['morphing level'] != 6)]
    return df_['rating'].values


def lds(css, gss) -> float:
    mean_css = np.mean(css)
    mean_gss = np.mean(gss)

    return round(mean_css-mean_gss, 3)


def lds_csv(path):
    lds_subjects = {}
    for file in glob.glob(path, recursive=True):
        sub_ = pd.read_csv(file).dropna()
        sub_id = sub_.iloc[0]['subject']
        css = extract_conditioned_stimuli_rating(sub_)
        gss = extract_generalization_stimuli(sub_)
        lds_subjects[sub_id] = lds(css, gss)

    ordered_dict = dict(sorted(lds_subjects.items()))
    df_= pd.DataFrame(list(ordered_dict.items()), columns=['subject', 'lds'])
    df_.to_csv('data/lds_subjects.csv', index=False)


if __name__ == '__main__':
    import os


    path = 'data/newLookAtMe**.csv'
    if not os.path.exists('data/lds_subjects.csv'):
        lds_csv(path)






