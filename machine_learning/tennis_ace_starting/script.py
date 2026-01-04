import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv("tennis_stats.csv")
print(df.head())
print(df.shape)
print(df.describe())
print(df.isnull().sum())
print(df.dtypes)


# perform exploratory analysis here:
features = ["Aces", "FirstServe", "BreakPointsOpportunities"]
outcomes = ["Wins", "Winnings", "Ranking"]

for feature in features:
    for outcome in outcomes:
        plt.scatter(df[feature], df[outcome])
        plt.xlabel(feature)
        plt.ylabel(outcome)
        plt.title(outcome + " vs " + feature)
        plt.show()

numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlations = numeric_df.corr()
print(correlations['Wins'].sort_values(ascending=False))


## perform single feature linear regressions here:
x = df[["BreakPointsOpportunities"]]
y = df[["Wins"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
plt.scatter(y_test, y_pred, alpha=0.4)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
score = regressor.score(x_test, y_test)
print(score)





## perform two feature linear regressions here:
x = df[['BreakPointsOpportunities', 'FirstServeReturnPointsWon']]
y = df[['Winnings']]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
plt.scatter(y_test, y_pred, alpha=0.4)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
score_2 = model.score(x_test, y_test)
print(score_2)


## perform multiple feature linear regressions here:
x = df[["Wins", "Aces", "BreakPointsOpportunities", "FirstServe", "FirstServeReturnPointsWon"]]
y = df[["Winnings"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
plt.scatter(y_test, y_pred, alpha=0.4)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
score_3 = model.score(x_test, y_test)
print(score_3)