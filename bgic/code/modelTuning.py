import pandas as pd
import xgboost as xgb
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


data_path = 'bgic/data/'
X_name = 'train_X.csv'
Y_name = ['test_Y_T20.csv', 'test_Y_T50.csv', 'test_Y_T80.csv', 'test_Y_D20.csv', 'test_Y_D50.csv', 'test_Y_D80.csv']

for y in Y_name:
    train_X = pd.read_csv(data_path + X_name)
    train_Y = pd.read_csv(data_path + y)
    train_X = train_X.loc[:1000]
    train_Y = train_Y.loc[:1000]

    # split data
    X_train, X_test, y_train, y_true = train_test_split(train_X, train_Y, test_size=0.2)
    # true_list = y_true.tolist()

    # parameter
    params = [1000, 1500, 2000, 2500, 3000, 3500, 4000]
    for param in params:
        model = model = xgb.XGBRegressor(n_estimators=param,
                             learning_rate=0.01,
                             max_depth=11,
                             min_child_weight=1,
                             gamma=0.05,
                             seed=0,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             reg_alpha=0,
                             reg_lambda=1)
        model.fit(X_train, y_train)
        y_pre = model.predict(X_test)
        # y_pre_list = y_pre.tolist()
        mse = mean_squared_error(y_true=y_true, y_pred=y_pre)
        rmse = math.sqrt(mse)
        print(y, ':', param, ':', rmse)
        pass
    pass

