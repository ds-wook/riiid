from cleaning.preprocessing import data_preprocessing
from sklearn.model_selection import train_test_split


path = '../../kaggle/input/riiid-test-answer-prediction/'
train, target = data_preprocessing(path)

X_train, X_test, y_train, y_test =\
    train_test_split(train, target, test_size=0.1, random_state=2020)
X_train, X_valid, y_train, y_valid =\
    train_test_split(X_train, y_train, random_state=2020, test_size=0.222)
