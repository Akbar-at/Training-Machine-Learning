from sklearn.datasets import make_circles
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
X,y = make_circles(noise=0.2,factor=0.5,random_state=1)
#print(y)
'''
plt.scatter(X[:,0],X[:,1],c = y)
plt.xlabel("X")
plt.ylabel("y")
plt.show()
'''
kf = KFold(n_splits=5, shuffle=True,random_state=1)
lr_scores=[]
rf_scores=[]
for train_in, test_in in kf.split(X):
    X_train, X_test = X[train_in],X[test_in]
    y_train, y_test = y[train_in],y[test_in]
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(X_train,y_train)
    lr_scores.append(lr.score(X_test,y_test))
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train,y_train)
    rf_scores.append(rf.score(X_test,y_test))

print("lr acc:",np.mean(lr_scores))
print("rf Acc:",np.mean(rf_scores))