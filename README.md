# kaggle
Record kaggle race thoughts and ideas

# 笔记
> 先撸一个baseline的model出来，再进行后续的分析步骤，一步步提高。后续步骤可能包括:<br/>
    1. 分析model现在的状态(欠/过拟合) (learning curve)<br/>
    2. 分析我们使用的feature的作用大小，进行feature selection<br/>
    3. 模型下的bad case和产生的原因<br/>
    4. 要做交叉验证

## 分析步骤

1. 认识数据
    - 数据较多很难看出规律，做图来看（matplotlib）
2. 对数据中的特殊点/离群点的分析和处理比较重要
3. 特征工程(feature engineering)太重要了！在很多Kaggle的场景下，甚至比model本身还要重要
4. 要做模型融合(model ensemble)
![avatar](https://github.com/huanbia/kaggle/blob/master/material/process.jpg)