import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score,recall_score,f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np


def specifity_score(y_true,y_pred): # make a function that returns the specifity score
    p,r,f,s = precision_recall_fscore_support(y_true,y_pred) #the fucntuon recision_recall_fscord [0] means r
    return r[0]
sensitivity_score = recall_score # state that the sensitivity score vlaue is the same as the recall score value
df = pd.read_csv('titanic.csv')


df['male'] = df['Sex'] == 'male'
kf = KFold(n_splits=5, shuffle = True) # uses the kfold class object, nsplits = 5 means there is 5 fold to the data , shuffle  true means randomize the order of data
X1 = df[['Fare','Pclass','male','Age','Siblings/Spouses','Parents/Children']].values
X2 = df[['Pclass','male','Age']].values
X3 = df[['Fare','Age']].values
# create 3 different X to measure the score of the model if there is less feature
y  = df['Survived'].values


scores = []


kf = KFold(n_splits = 5, shuffle = True)
def Get_Model_Score (X,y,kf): # creates a function to compare 3 models with a measurements of Accuracy, precision,recall, and f1 score
    Acc_scores = []
    Prec_scores = []
    recall_scores = []
    f1_scores = []
    for train_in, test_in in kf.split(X):
        X_train,X_test = X[train_in], X[test_in]
        y_train,y_test= y[train_in], y[test_in]
        model = LogisticRegression(solver='liblinear')
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        Acc_scores.append(accuracy_score(y_test, y_pred))
        Prec_scores.append(precision_score(y_test,y_pred))
        recall_scores.append(recall_score(y_test,y_pred))
        f1_scores.append(f1_score(y_test,y_pred))
    print("accuracy:", np.mean(Acc_scores))
    print("precision:", np.mean(Prec_scores))
    print("recall:", np.mean(recall_scores))
    print("f1 score:", np.mean(f1_scores))



print ("First Model")
Get_Model_Score(X1,y,kf)


print("Second Model")
Get_Model_Score(X2,y,kf)

print("Third Model")
Get_Model_Score(X3,y,kf)
