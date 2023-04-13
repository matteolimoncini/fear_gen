import extract_correct_csv
import pandas as pd
import os
# parent folder in order to read correct file
os.chdir('..')

list_subject = extract_correct_csv.extract_only_valid_subject()

appended_df = []
for x in list_subject:
    cur_sub = extract_correct_csv.read_correct_subject_csv(x)
    df_ = pd.read_csv('data/LookAtMe_0'+cur_sub+'.csv', sep='\t')
    df_ = df_[['subject', 'trial', 'rating', 'shock']]
    appended_df.append(df_)
appended_df = pd.concat(appended_df)
appended_df.to_csv('csv_shock.csv')
