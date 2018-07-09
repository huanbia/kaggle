# encoding=utf-8

import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import accuracy_score
from utils.ModelEvaluation import plot_learning_curve
import xgboost as xgb
from sklearn.model_selection import ShuffleSplit, KFold
from utils.DataRepair import repair

events_cols = []
DATA_DIR = "data"
OUT_DATA_DIR = "output"


def strcol2numcol(inputCol):
    ''' convert factor/character to numeric
    :param inputCol: factor/character column
    :return: the numeric column
    '''
    sort_unique = inputCol[inputCol.notnull()].sort_values().unique()
    inputCol = inputCol.replace(sort_unique, range(1, len(sort_unique) + 1))
    return inputCol


def data_process(input_data, age_mean, fare_mean):
    input_data["Sex"] = strcol2numcol(input_data["Sex"])
    input_data[input_data["Age"] < 1]["Age"] = 1
    input_data["Age"] = input_data["Age"].replace([np.nan], int(age_mean))
    input_data["Fare"] = input_data["Fare"].replace([np.nan], fare_mean)
    input_data["Embarked"] = input_data["Embarked"].replace([np.nan], "Q")
    input_data["Embarked"] = strcol2numcol(input_data["Embarked"])
    # input_data = repair(input_data)
    return input_data


def train_operate():
    '''
    :return: 对训练集进行操作
    这时候我们通常不再划分一个测试集，可能的原因有两个：1、比赛方基本都很抠，训练集的样本本来就少；2、我们也没法保证要提交的测试集是否跟训练集完全同分布，因此再划分一个跟训练集同分布的测试集就没多大意义了。
    '''
    allcols = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
               'Fare', 'Survived']
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked','Survived']
    features1 = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    # features1 = ['Pclass', 'Sex','SibSp']
    # 训练集 (891,12)
    train = pd.read_csv("/".join([DATA_DIR, "train.csv"]), sep=",")
    # 测试集 (418, 11)
    test = pd.read_csv("/".join([DATA_DIR, "test.csv"]), sep=",")

    submission = pd.read_csv("/".join([DATA_DIR, "gender_submission.csv"]), sep=",")
    test = pd.merge(test, submission, how="inner", on="PassengerId")
    train_data = train[features].copy()
    test_data = test[features].copy()
    age_mean = train_data["Age"].mean()
    fare_mean = train_data["Fare"].mean()
    train_data = data_process(train_data, age_mean, fare_mean)
    test_data = data_process(test_data, age_mean, fare_mean)

    target = 'Survived'
    print("[Info] Start process training data...")

    # 利用随机森林进行预测
    print("[Info] Start Training...")
    start = time.time()
    rfc = RandomForestClassifier(random_state=100)
    # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = KFold(10, shuffle=True)
    # plt = plot_learning_curve(rfc,"test",train[features1],train[target],cv=cv)
    # return plt

    rfc.fit(train_data[features1], train_data[target])
    predict_result = rfc.predict(test_data[features1])
    accuracy = accuracy_score(test_data[target], predict_result)
    print("[Info] End Training, time is %fs" % (time.time() - start))
    print("accuracy is %f" % accuracy)
    test_data["predicted"] = predict_result
    test = pd.DataFrame(data = {"PassengerId":test["PassengerId"], "Survived":test_data["predicted"]})
    # test = test.rename(columns={"predicted": "Survived"})
    test.to_csv("/".join([OUT_DATA_DIR, time.strftime("%Y%m%d%H%M%S", time.localtime()) + ".csv"]), index=False)


if __name__ == "__main__":
    plt = train_operate()
    # plt.show()