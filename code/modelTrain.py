import pandas as pd
import xgboost as xgb
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


data_path = 'Kaggle/data/'
model_path = 'Kaggle/model/'
input_data_name = 'input.csv'

input_data = pd.read_csv(data_path + input_data_name)
drop_columns = ['TotalTimeStopped_p20',
                'TotalTimeStopped_p50',
                'TotalTimeStopped_p80',
                'DistanceToFirstStop_p20',
                'DistanceToFirstStop_p50',
                'DistanceToFirstStop_p80']

train_X = input_data.drop(columns=drop_columns, axis=1)
train_Y_T20 = input_data.loc[:, 'TotalTimeStopped_p20']
train_Y_T50 = input_data.loc[:, 'TotalTimeStopped_p50']
train_Y_T80 = input_data.loc[:, 'TotalTimeStopped_p80']
train_Y_D20 = input_data.loc[:, 'DistanceToFirstStop_p20']
train_Y_D50 = input_data.loc[:, 'DistanceToFirstStop_p50']
train_Y_D80 = input_data.loc[:, 'DistanceToFirstStop_p80']


# # test
train_X = train_X.loc[:100]
train_Y_T20 = train_Y_T20.loc[:100]
train_Y_T50 = train_Y_T50.loc[:100]
train_Y_T80 = train_Y_T80.loc[:100]
train_Y_D20 = train_Y_D20.loc[:100]
train_Y_D50 = train_Y_D50.loc[:100]
train_Y_D80 = train_Y_D80.loc[:100]

X_train_T20, X_test_T20, y_train_T20, y_true_T20 = train_test_split(train_X, train_Y_T20, test_size=0.2)
X_train_T50, X_test_T50, y_train_T50, y_true_T50 = train_test_split(train_X, train_Y_T50, test_size=0.2)
X_train_T80, X_test_T80, y_train_T80, y_true_T80 = train_test_split(train_X, train_Y_T80, test_size=0.2)

X_train_D20, X_test_D20, y_train_D20, y_true_D20 = train_test_split(train_X, train_Y_D20, test_size=0.2)
X_train_D50, X_test_D50, y_train_D50, y_true_D50 = train_test_split(train_X, train_Y_D50, test_size=0.2)
X_train_D80, X_test_D80, y_train_D80, y_true_D80 = train_test_split(train_X, train_Y_D80, test_size=0.2)

true_list_T20 = y_true_T20.tolist()
true_list_T50 = y_true_T50.tolist()
true_list_T80 = y_true_T80.tolist()

true_list_D20 = y_true_D20.tolist()
true_list_D50 = y_true_D50.tolist()
true_list_D80 = y_true_D80.tolist()


params = [1800, 2000, 2200, 2500, 2800, 3000, 3500]
for param in params:
    model = xgb.XGBRegressor(n_estimators=param,
                             learning_rate=0.1,
                             max_depth=11,
                             min_child_weight=1,
                             gamma=0.05,
                             seed=0,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             reg_alpha=0,
                             reg_lambda=1
                             )
    model.fit(X_train_T20, y_train_T20)
    y_pre_T20 = model.predict(X_test_T20)
    model.get_booster().save_model(model_path + str(param) + '_T20_.model')
    y_pre_list_T20  = y_pre_T20.tolist()

    mse = mean_squared_error(y_true=y_true_T20, y_pred=y_pre_list_T20)
    rmse = math.sqrt(mse)
    print(param)
    print(rmse)
