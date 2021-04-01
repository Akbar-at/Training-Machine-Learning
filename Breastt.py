import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt


cancer_data = load_breast_cancer()


print(cancer_data.keys())# printing keys to help understand datasets and commands
df = pd.DataFrame(cancer_data['data'], columns = cancer_data['feature_names'])
# code above is creating a dat frame with keys obtainded from .keys(). 'data' is the actual data and the columns are obtained from the 'feature_names'_)
#print(cancer_data['target_names'])#the result is ['malignant' 'bening'] the first one is 0 meaning false and the rest follows
df['target'] = cancer_data['target'] #adding the target into the data frame


'''param_grid= {
    'n_estimators': (10),
}'''
X = df[cancer_data['feature_names']].values
y = df['target'].values
rf = RandomForestClassifier(n_estimators=10, random_state=12)
#gs = GridSearchCV(rf,param_grid,scoring='f1',cv=5)
#print(rf.score(X_test,y_test)
worst_cols = [col for col in df.columns if 'worst' in col]
print(worst_cols)
X_worst = df[worst_cols]
X_train, X_test, y_train, y_test = train_test_split(X_worst,y,random_state=29)

rf.fit(X_train,y_train)
print(rf.score(X_test,y_test))

X_tr, X_te, y_tr, y_te = train_test_split(X,y,random_state=29)
rf.fit(X_tr,y_tr)
print(rf.score(X_te,y_te))
''''
rf.fit(X_train,y_train)
ft_imp = pd.Series(rf.feature_importances_,
index = cancer_data.feature_names).sort_values(ascending=False)
print(ft_imp.head(10))

this bundle of code finds the importances of feature
'''




plot(n_estimators,scores)
plt.xlabel("n_estimators")
plt.ylabel("accuracy")
plt.xlim(0,100)
plt.ylim(0.9,1)
plt.show()
'''

print('rfscore', rf.score(X_test,y_test))


dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
print("dt score: ", dt.score(X_test,y_test))
'''
#print ((Bruh == Y).sum()/Y.shape[0])
#print(model.score(X,Y))
'''

