import pandas as pd  # loading and preprocessing

import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif

###################################################################################################################

print("Loading CSV")
dataframe = pd.read_csv("full.csv") # df = dataframe

print(dataframe.isnull().sum())

dropped_cols = ["PassengerId", "Name", "Ticket", "Cabin", "Hometown", "Destination", "WikiId", "Age", "Lifeboat", "Body", "Embarked"]
dataframe = dataframe.drop(dropped_cols, axis=1) # drop all columns (irrelevant features) which are store in the array

print("Preprocessing data\n")
dataframe["Fare"] = dataframe["Fare"].fillna(dataframe["Fare"].median()) # median/mean for numerical values 
dataframe["Age_wiki"] = dataframe["Age_wiki"].fillna(dataframe["Age_wiki"].median()) 

labelEncoder = LabelEncoder()
dataframe["Sex"] = dataframe["Sex"].map({"male": 1, "female": 0}) # set the male and female values to numerical

def titles(row):
    if (pd.notna(row["Name_wiki"])):
        name = str(row["Name_wiki"]).lower()
        
        marriage_status = 0
        rank = 0
        nobility = 0
        
        for title in ["mrs", "mme"]:
            if title in name:
                marriage_status = 1
                break
        for title in ["col", "major", "capt", "dr", "rev", "father"]:
            if title in name:
                rank = 1
                break
        for title in ["jonkheer", "sir", "countess", "don"]:
            if title in name:
                nobility = 1
                break
        
        return pd.Series([marriage_status, rank, nobility])

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html
dataframe[["Marriage_status", "Rank", "Nobility"]] = dataframe.apply(titles, axis=1)
dataframe.pop('Name_wiki')

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html
dataframe.loc[dataframe["Age_wiki"] <= 17, "Youth"] = 1
dataframe.loc[dataframe["Age_wiki"] > 17, "Youth"] = 0
dataframe["Boarded"] = dataframe["Boarded"].map({"Cherbourg": 3, "Southampton": 2, "Queenstown": 1})

dataframe = dataframe.dropna(subset=["Boarded"])
dataframe = dataframe.dropna(subset=["Survived"])

features = ["Pclass", "Sex", "Fare", "Age_wiki", "SibSp", "Parch", "Boarded", "Marriage_status", "Rank", "Nobility", "Youth"]

features_train, features_test, target_train, target_test = train_test_split(dataframe[features], dataframe["Survived"], test_size=0.2, random_state=42)

###################################################################################################################

metrics = []
modScores = []

print("Training with Random Forest")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(features_train, target_train)
rf_predictions = rf_model.predict(features_test)

accuracy = accuracy_score(target_test, rf_predictions) * 100
print(f"Random Forest accuracy: {accuracy:.2f}%")
modScores.append(accuracy)
metrics.append(f"Accuracy ({accuracy:.2f}%)")

precision = precision_score(target_test, rf_predictions) * 100
print(f"Random Forest precision: {precision:.2f}%")
modScores.append(precision)
metrics.append(f"Precision ({precision:.2f}%)")

recall = recall_score(target_test, rf_predictions) * 100
print(f"Random Forest recall: {recall:.2f}%")
modScores.append(recall)
metrics.append(f"Recall ({recall:.2f}%)")

f1 = f1_score(target_test, rf_predictions) * 100
print(f"Random Forest F1: {f1:.2f}%")
modScores.append(f1)
metrics.append(f"F1 Score ({f1:.2f}%)")

rf_cm = confusion_matrix(target_test, rf_predictions)
print(f"Random Forest confusion matrix:\n{rf_cm}\n")

probabilities = rf_model.predict_proba(features_test)[:, 1]
auc = roc_auc_score(target_test, probabilities)
auc = auc * 100
print(f"AUC score: {auc:.2f}%")
modScores.append(auc)
metrics.append(f"AUC Score ({auc:.2f}%)")

plt.figure(figsize=(9, 6))
plt.bar(metrics, modScores, color=["green", "red", "orange", "blue", "black", "yellow"])
plt.title("RF Overall Metrics without Highest Gini Scored Features")
plt.xlabel("Metric")
plt.ylabel("Score (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()


# Gini-based feature importances
importances = rf_model.feature_importances_
print(f"Random Forest Feature Importances (Gini-based): {importances}\n")

plt.figure(figsize=(9, 6))
plt.bar(features, importances, color="blue")
plt.title("RF Gini Importance Scores")
plt.xlabel("Features")
plt.ylabel("Raw Gini Score (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

###################################################################################################################

print("Training SVM on individual features")
for feature in features:
    print("Training with:", feature)

    correlation = dataframe[feature].corr(dataframe["Survived"])
    print(f"Correlation between {feature} and Survival: {correlation:.2f}")

    feature_train = features_train[[feature]]
    feature_test = features_test[[feature]]

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(feature_train, target_train)
    rf_predictions = rf_model.predict(feature_test)

    rf_accuracy = accuracy_score(target_test, rf_predictions) * 100
    print(f"Random Forest accuracy: {rf_accuracy:.2f}%")

    rf_precision = precision_score(target_test, rf_predictions) * 100
    print(f"Random Forest precision: {rf_precision:.2f}%")

    rf_recall = recall_score(target_test, rf_predictions) * 100
    print(f"Random Forest recall: {rf_recall:.2f}%")

    rf_f1 = f1_score(target_test, rf_predictions) * 100
    print(f"Random Forest F1: {rf_f1:.2f}%")

    rf_cm = confusion_matrix(target_test, rf_predictions)
    print(f"Random Forest confusion matrix:\n{rf_cm}\n")