import pandas as pd
import xgboost as xgb
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


data_path = 'Kaggle/data/'
model_path = 'Kaggle/model/'
X_name = 'train_X.csv'
Y_name = ['test_Y_T20.csv', 'test_Y_T50.csv', 'test_Y_T80.csv', 'test_Y_D20.csv', 'test_Y_D50.csv', 'test_Y_D80.csv']

for y in Y_name:
    train_X = pd.read_csv(data_path + X_name)
    train_Y = pd.read_csv(data_path + y)
    # split data
    X_train, X_test, y_train, y_true = train_test_split(train_X, train_Y, test_size=0.2)
    true_list = y_true.tolist()

    # parameter
    params = [1800]
    for param in params:
        model = xgb.Booster(model_file=model_path + '2800_T20_.model')
        y_pre = model.predict(X_test)
        y_pre_list = y_pre.tolist()
        mse = mean_squared_error(y_true=y_true, y_pred=y_pre_list)
        rmse = math.sqrt(mse)
        print(param)
        print(rmse)
        pass
    pass

