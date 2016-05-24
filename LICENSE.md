import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import matplotlib as mlp
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
%matplotlib inline
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

df=pd.read_csv('German_credit_Training2.csv')
dummies = pd.get_dummies(df['Account.balance'])
