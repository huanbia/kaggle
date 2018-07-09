# encoding = utf-8
# 首先找出数据中哪些列是有缺失值，哪些列是没有缺失值的
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# input_data = pd.read_csv("../data/train.csv")

def repair(input_data):
    null_sum = input_data.isnull().sum()
    features = []
    for i in range(len(null_sum)):
        if null_sum[i] == 0:
            features.append(null_sum.index[i])
    predicts = [x for x in input_data.columns if x not in features]
    for predict in predicts:
        train = input_data[input_data[predict].notnull()]
        test = input_data[input_data[predict].isnull()]
        if len(list(input_data[predict].drop_duplicates())) / input_data[predict].shape[0] > 0.01:  # using regression model
            fr = RandomForestRegressor()
            fr.fit(train[features], train[predict])
            pr_result = fr.predict(test[features])
            input_data.loc[input_data[predict].isnull(),predict] = pr_result
        else:
            fc = RandomForestClassifier()
            fc.fit(train[features], train[predict])
            pr_result = fc.predict(test[features])
            input_data.loc[input_data[predict].isnull(), predict] = pr_result

    return input_data