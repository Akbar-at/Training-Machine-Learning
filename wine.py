import pandas as pd
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression

wine_data = load_wine()

print(wine_data.keys())

df = pd.DataFrame(wine_data.data , columns = wine_data.feature_names)

df['class'] = wine_data.target
X = df[wine_data.feature_names].values
Y = df['class'].values

model = LogisticRegression(solver='liblinear')
model.fit(X,Y)

Bruh = model.predict(X)

print((Bruh == Y).sum())
print(model.score(X,Y))
print(Y.shape[0])
print(model.coef_ , model.intercept_)

