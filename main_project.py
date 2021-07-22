# Import relevant libraries for data visualization and processing
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

# importing dataset and assigning it to 'df'
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Making all columns visible in Pycharm console
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 35)

# REGEX DUMMY EXAMPLE. '\D' finds and prints all single non digit characters in sentence
print(re.findall('\D', 'I9 8L03oV8555e 9R093e55G88e12348X79'))

# print head of dataset
print(df.head)

# print info of dataset
print(df.info())

# Checking Dataframe for duplicate values
print("This Dataframe contains ", df.duplicated().sum(), " duplicates.")

# check to see if there are any null values in dataset. returns a boolean value
print("Does this Dataframe contain any null values : ", df.isnull().values.any())

# print statistical data of dataset
print(df.describe())

# print number of employees who have stayed and left the company
print(df['Attrition'].value_counts())

# countplot to visualize Attrition data
sns.countplot(x='Attrition', data=df, palette="Set1")
plt.show()


# function that reads in graph characteristics and returns a countplot
def plots(x_axis, hue_1, data_1):
    plt.figure(figsize=(11, 5))
    sns.countplot(x=x_axis, hue=hue_1, data=data_1, palette="Set1")
    return plt.show()


#
plots('Age', 'Attrition', df)
plots('Gender', 'Attrition', df)

# Dropping of irrelevant columns from dataset
df.drop('EmployeeCount', axis=1, inplace=True)
df.drop('StandardHours', axis=1, inplace=True)
df.drop('EmployeeNumber', axis=1, inplace=True)
df.drop('Over18', axis=1, inplace=True)

# print shape of dataset to confirm columns have been dropped
print(df.shape)  # 1470, 31

# visualise the correlation for each column
plt.figure(figsize=(12, 12))
sns.heatmap(df.corr(), annot=True, fmt='.0%')
plt.show()

# Converting target column to a numerical data type
df['Attrition'] = df.Attrition.astype("category").cat.codes
print(df.Attrition.value_counts())

# creating categorical variables through One Hot Encoding and dropping irrelevant columns
training = pd.get_dummies(df, columns=["BusinessTravel", "Department", "EducationField",
                                       "Gender", "JobRole", "MaritalStatus", "OverTime"])
print(training.columns)
training.drop("Gender_Female", axis=1, inplace=True)
training.drop("OverTime_No", axis=1, inplace=True)

# creating sets of independent and dependent variables
X = training.drop("Attrition", axis=1)
y = training["Attrition"]
print(X.shape)
print(y.shape)

# splitting into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Initiating RandomForestClassifier with 10 estimators and criterion set to entropy
clf = RandomForestClassifier(n_estimators=10, criterion='entropy')
clf = clf.fit(X_train, y_train)

# Make predictions on test set for evaluating the model
predictions = clf.predict(X_test)

# Printing the Accuracy, Precision, Recall and F1 Score of the RandomForest Classifier Model
print("Model     : Random Forest Classifier")
print("Accuracy  : ", metrics.accuracy_score(y_test, predictions))
print("Precision : ", metrics.precision_score(y_test, predictions))
print("Recall    : ", metrics.recall_score(y_test, predictions))
print("F1 Score  : ", metrics.f1_score(y_test, predictions))

# Random Forest Confusion Matrix
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

# Random Forest Confusion Matrix Heatmap
plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actuals')
plt.xlabel('Predictions')
all_sample_title = 'RandomForest Confusion Matrix\nAccuracy Score: {0}'.format(
    metrics.accuracy_score(y_test, predictions))
plt.title(all_sample_title, size=15)
plt.show()

# RandomForest Feature importance
Feat_Imp = pd.DataFrame({"Features": X_train.columns, "Importance": clf.feature_importances_})
Feat_Imp = Feat_Imp.sort_values(by=['Importance'], ascending=False)
print(Feat_Imp)
#
plt.figure(figsize=(20, 40))
plt.title('Random forest Feature Importance', size=10)
sns.barplot(x="Importance", y="Features", data=Feat_Imp)
plt.show()

# Initiating GradientBoosting Classifier
gb = GradientBoostingClassifier()

# Fitting GradientBoosting Classifier to training datasets
gb.fit(X_train, y_train)

# Make predictions on test set for evaluating the model
y_pred = gb.predict(X_test)

# GradientBoosting Confusion Matrix
gb_cm = metrics.confusion_matrix(y_test, y_pred)
print(gb_cm)

# initiating confusion matrix variables for manual calculations of model performance
true_negative = gb_cm[0][0]
false_positive = gb_cm[0][1]
false_negative = gb_cm[1][0]
true_positive = gb_cm[1][1]

# Manually calculating accuracy of GB model
Accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print("Model     : Gradient Boosting Classifier")
print("Accuracy  : ", Accuracy)

# Manually calculating precision of GB model
Precision = true_positive / (true_positive + false_positive)
print("Precision : ", Precision)

# Manually calculating Recall of GB model
Recall = true_positive / (true_positive + false_negative)
print("Recall    : ", Recall)

# Manually calculating F1 Score of GB model
F1_Score = 2 * (Recall * Precision) / (Recall + Precision)
print("F1 Score  : ", F1_Score)

# GradientBoosting Confusion Matrix Heatmap
plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Reds_r')
plt.ylabel('Actuals')
plt.xlabel('Predictions')
all_sample_title = 'GradientBoosting Confusion Matrix\nAccuracy Score: {0}'.format(
    metrics.accuracy_score(y_test, y_pred))
plt.title(all_sample_title, size=15)
plt.show()

# GradientBoosting Feature importance
Feat_Imp = pd.DataFrame({"Features": X_train.columns, "Importance": gb.feature_importances_})
Feat_Imp = Feat_Imp.sort_values(by=['Importance'], ascending=False)
print(Feat_Imp)

# Visualizing Feature Importance results from GradientBoosting
plt.figure(figsize=(20, 40))
plt.title('Gradient Boosting Feature Importance', size=10)
sns.barplot(x="Importance", y="Features", data=Feat_Imp)
plt.show()
