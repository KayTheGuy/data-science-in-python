import sys
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def get_data():
    """
        Read training and test data from csv files. Read output filename
        Return features(X), labels(y), data to be predicted, and output filename
    """
    if len(sys.argv) >= 4:
        data = pd.read_csv(sys.argv[1])
        data_to_predict = pd.read_csv(sys.argv[2])
        output_filename = sys.argv[3]
        # all columns except city as features
        X = data.ix[:, data.columns != 'city'].values 
        X_to_predict = data_to_predict.ix[:, data_to_predict.columns != 'city'].values 
        # city as labels
        y = data['city'].values
        return X, y, X_to_predict, output_filename
    else:
        print('Unable to read the csv files and output filename: \
        provide a valid csv file names in the same directory and output csv filename!')
        sys.exit()


def output_labels(predictions, filename):
    pd.Series(predictions).to_csv(filename, index=False)

def get_svc_score_predictions(X_train, y_train, X_test, y_test, X_to_predict):
    """
        Read training and test data from csv files. Read output filename
        Return features(X), labels(y), data to be predicted, and output filename
    """
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel='linear', C=0.2)
    )
    model.fit(X_train, y_train)
    return model.score(X_test, y_test), model.predict(X_to_predict)

def main():
    X, y, X_to_predict, output_filename = get_data()

    # split data into training and test data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    score, predictions = get_svc_score_predictions(X_train, y_train, X_test, y_test, X_to_predict)
    print (score)
    output_labels(predictions, output_filename)

if __name__ == '__main__':
    main()
