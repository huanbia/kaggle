# encoding=utf-8

import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import accuracy_score
import xgboost as xgb

events_cols = []
DATA_DIR = "data"
OUT_DATA_DIR = "output"


def strcol2numcol(inputCol):
    ''' convert factor/character to numeric
    :param inputCol: factor/character column
    :return: the numeric column
    '''
    sort_unique = inputCol.sort_values().unique()
    inputCol = inputCol.replace(sort_unique, range(1, len(sort_unique) + 1))
    return inputCol


def data_process(input_data, age_mean, fare_mean):
    input_data["Sex"] = strcol2numcol(input_data["Sex"])
    input_data[input_data["Age"] < 1]["Age"] = 1
    input_data["Age"] = input_data["Age"].replace([np.nan], int(age_mean))
    input_data["Fare"] = input_data["Fare"].replace([np.nan], fare_mean)
    input_data["Embarked"] = input_data["Embarked"].replace([np.nan], "Q")
    input_data["Embarked"] = strcol2numcol(input_data["Embarked"])
    return input_data


def train_operate():
    '''
    :return: 对训练集进行操作
    '''
    allcols = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
               'Fare', 'Survived']
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
    # 训练集 (891,12)
    train = pd.read_csv("/".join([DATA_DIR, "train.csv"]), sep=",")
    # 测试集 (418, 11)
    test = pd.read_csv("/".join([DATA_DIR, "test.csv"]), sep=",")

    submission = pd.read_csv("/".join([DATA_DIR, "gender_submission.csv"]), sep=",")

    test = pd.merge(test, submission, how="inner", on="PassengerId")
    train = train[features]
    # test = test[features]
    age_mean = train["Age"].mean()
    fare_mean = train["Fare"].mean()
    train = data_process(train, age_mean, fare_mean)
    test = data_process(test, age_mean, fare_mean)

    target = 'Survived'
    print("[Info] Start process training data...")
    start = time.time()

    # 利用随机森林进行预测
    print("[Info] Start Training...")
    start = time.time()
    rfc = RandomForestClassifier(random_state=100)
    rfc.fit(train[features], train[target])
    predict_result = rfc.predict(test[features])
    accuracy = accuracy_score(test[target], predict_result)
    print("[Info] End Training, time is %fs" % (time.time() - start))
    print("accuracy is %f" % accuracy)
    test["predicted"] = predict_result
    test = test[["PassengerId", "predicted"]]
    test = test.rename(columns={"predicted": "Survived"})
    test.to_csv("/".join([OUT_DATA_DIR, time.strftime("%Y%m%d%H%M%S", time.localtime()) + ".csv"]), index=False)


if __name__ == "__main__":
    train_operate()
