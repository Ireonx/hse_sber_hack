import joblib
import pandas as pd


def main():
    test_data = pd.read_csv('data/predict.csv', index_col='client_id')
    model = joblib.load(open('model.pkl', 'rb'))
    output = model.predict_proba(test_data)
    submission = pd.read_csv('data/test.csv')
    submission['probability'] = output[:,1]
    submission.drop('Unnamed: 0', axis=1, inplace=True)
    submission.to_csv('result.csv')

main()