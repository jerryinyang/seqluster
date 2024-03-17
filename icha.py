import pandas as pd

data = pd.read_csv("/Users/jerryinyang/Code/seqluster/CITIZEN QUAESTIONNAIRE  .csv")
clean_data = data.drop(['Timestamp', 'Username', 'QUESTIONNAIRE CONSENT FORM', 'Unnamed: 28'], axis=1)

for column in clean_data.columns:
    print(clean_data[column].value_counts())
    input()