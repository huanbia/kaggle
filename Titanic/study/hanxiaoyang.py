# encoding = utf-8
import matplotlib.pyplot as plt
import pandas as pd


data_train = pd.read_csv("../data/train.csv")
data_train.Survived.value_counts().plot(kind='bar')