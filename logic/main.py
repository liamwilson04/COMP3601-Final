import pandas as pd  # loading and preprocessing

import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# evaluation metrics, acc, cm, f1, (precision and recall if comparing models)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance # this might be removed

flag = True

while flag:
    func = input("Enter function (1: Models, 2: Weka): ")
    if (func.lower() == "1"):
        model = input("Enter model (SVM, RF): ")
        if (model.lower() == "svm" or model.lower() == "rf"):
            flag = False
        else:
            print("Invalid model. Please try again.")

    if func.lower() == "2":
        flag = False
    
    else:
        print("Invalid function. Please try again.")
    
        

#Load CSV and display some info
print("Loading CSV")
dataframe = pd.read_csv("full.csv") # df = dataframe

