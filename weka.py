import pandas as pd # loading and preprocessing
import seaborn as sb # visualization

# more preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# actual model
from sklearn.svm import SVC

# evaluation metrics, acc, cm, f1, (precision and recall if comparing models)
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance # this might be removed

############################################################################################################################################################################

#Load CSV and display some info
print("Loading CSV")
dataframe = pd.read_csv("full.csv") # df = dataframe

dropped_cols = ["PassengerId", "Name", "Ticket", "Cabin", "Hometown", "Destination", "WikiId", "Age", "Lifeboat", "Body", "Class", "Embarked"]
dataframe = dataframe.drop(dropped_cols, axis=1) # drop all columns (irrelevant features) which are store in the array

print("Preprocessing data")
dataframe["Fare"] = dataframe["Fare"].fillna(dataframe["Fare"].median()) # median/mean for numerical values 
dataframe["Age_wiki"] = dataframe["Age_wiki"].fillna(dataframe["Age_wiki"].median()) 

dataframe = dataframe.dropna(subset=["Survived"])
dataframe = dataframe.dropna(subset=["Boarded"])

def titles(row):
    if (pd.notna(row["Name_wiki"])):
        name = str(row["Name_wiki"]).lower()
        
        marriage_status = "No"
        rank = "No"
        nobility = "No"
        
        for title in ["mrs", "mme"]:
            if title in name:
                marriage_status = "Yes"
                break
        for title in ["col", "major", "capt", "dr", "rev", "father"]:
            if title in name:
                rank = "Yes"
                break
        for title in ["jonkheer", "sir", "countess", "don"]:
            if title in name:
                nobility = "Yes"
                break
        
        return pd.Series([marriage_status, rank, nobility])

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html
dataframe[["Marriage_status", "Rank", "Nobility"]] = dataframe.apply(titles, axis=1)
dataframe.pop('Name_wiki')

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html
dataframe.loc[dataframe["Age_wiki"] <= 17, "Youth"] = "Yes"
dataframe.loc[dataframe["Age_wiki"] > 17, "Youth"] = "No"

surv = dataframe.pop('Survived')
dataframe['Survived'] = surv

dataframe["Boarded"] = dataframe["Boarded"].map({"Cherbourg": 3, "Southampton": 2, "Queenstown": 1})
dataframe["Survived"] = dataframe["Survived"].map({1: "Yes", 0: "No"})

dataframe.to_csv("preprocessed_titanic.csv", index=False)