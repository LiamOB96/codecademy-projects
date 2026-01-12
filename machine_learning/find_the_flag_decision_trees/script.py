import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data
cols = ['name','landmass','zone', 'area', 'population', 'language','religion','bars','stripes','colours',
'red','green','blue','gold','white','black','orange','mainhue','circles',
'crosses','saltires','quarters','sunstars','crescent','triangle','icon','animate','text','topleft','botright']
df= pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data", names = cols)

#variable names to use as predictors
var = [ 'red', 'green', 'blue','gold', 'white', 'black', 'orange', 'mainhue','bars','stripes', 'circles','crosses', 'saltires','quarters','sunstars','triangle','animate']

#Print number of countries by landmass, or continent
print(df["landmass"].value_counts())

#Create a new dataframe with only flags from Europe and Oceania
eur_ocea_df = df[(df["landmass"] == 3) | (df["landmass"] == 6)]

#Print the average values of the predictors for Europe and Oceania
numeric_vars = eur_ocea_df[var].select_dtypes(include="number").columns
print(eur_ocea_df.groupby("landmass")[numeric_vars].mean())

#Create labels for only Europe and Oceania
labels = (eur_ocea_df["landmass"] == 6).astype(int)
# Oceania = 1, Europe = 0

#Print the variable types for the predictors
print(eur_ocea_df[var].dtypes)

#Create dummy variables for categorical predictors
data = pd.get_dummies(eur_ocea_df[var])

#Split data into a train and test set
X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state = 1, test_size = 0.2)

#Fit a decision tree for max_depth values 1-20; save the accuracy score in acc_depth
acc_depth = []

for depth in range(1, 21):
  classifier = DecisionTreeClassifier(max_depth = depth)
  classifier.fit(X_train, y_train)
  acc_depth.append(classifier.score(X_test, y_test))


#Plot the accuracy vs depth
depths = range(1, 21)
plt.plot(depths, acc_depth)
plt.xlabel("max_depth")
plt.ylabel("Test Accuracy")
plt.title("Decision Tree Accuracy vs max_depth")
plt.show()
plt.close()

#Find the largest accuracy and the depth this occurs
largest_acc = max(acc_depth)
largest_acc_depth = acc_depth.index(largest_acc) + 1
print(f'Highest accuracy {round(largest_acc,3)} at depth {largest_acc_depth}')

#Refit decision tree model with the highest accuracy and plot the decision tree
refit_clf = DecisionTreeClassifier(max_depth=largest_acc_depth, random_state=42)
refit_clf.fit(X_train, y_train)

plt.figure(figsize=(14,8))
tree.plot_tree(
    refit_clf,
    feature_names=data.columns,
    class_names=["Europe", "Oceania"],
    filled=True,
)
plt.show()
plt.close()
#Create a new list for the accuracy values of a pruned decision tree.  Loop through
#the values of ccp and append the scores to the list
acc_pruned = []

ccp = np.logspace(-3, 0, num=20)
for alpha in ccp:
    classifier = DecisionTreeClassifier(random_state = 1, max_depth = largest_acc_depth, ccp_alpha=alpha)
    classifier.fit(X_train, y_train)
    acc_pruned.append(classifier.score(X_test, y_test))

#Plot the accuracy vs ccp_alpha
plt.plot(ccp, acc_pruned)
plt.xlabel("ccp_alpha")
plt.ylabel("Test Accuracy")
plt.title("Decision Tree Accuracy vs ccp_alpha")
plt.show()
plt.close()

#Find the largest accuracy and the ccp value this occurs
best_acc = max(acc_pruned)
best_ccp_index = acc_pruned.index(best_acc)
best_ccp_alpha = ccp[best_ccp_index]

print(f'Highest accuracy {round(best_acc,3)} at ccp_alpha {best_ccp_alpha}')

#Fit a decision tree model with the values for max_depth and ccp_alpha found above
final_clf = DecisionTreeClassifier(max_depth=largest_acc_depth, random_state=42, ccp_alpha = best_ccp_alpha)
final_clf.fit(X_train, y_train)

#Plot the final decision tree
plt.figure(figsize=(14,8))
tree.plot_tree(
    final_clf,
    feature_names=data.columns,
    class_names=["Europe", "Oceania"],
    filled=True,
)
plt.show()
plt.close()