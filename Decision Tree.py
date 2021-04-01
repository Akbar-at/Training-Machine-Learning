#DecisionTreeClassifier Class
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score,recall_score,f1_score, roc_curve, roc_auc_score
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import export_graphviz
import graphviz
from IPython.display import Image
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz 2.45.20200729.1351\bin'


df =  pd.read_csv('titanic.csv')
df['male'] = df ['Sex'] == 'male'
feature_names = ['Fare','Pclass','male','Age']
X = df[feature_names].values
y = df['Survived'].values

param_grid = {
    'max_depth':[5,15,25],
    'min_samples_leaf':[1,3],
    'max_leaf_nodes':[10,20,35,50]
} # set a pram grid variable to find the best pruning method

dt = DecisionTreeClassifier(max_depth= 25 ,max_leaf_nodes= 35, min_samples_leaf=3)
#,X_test,y_train,y_test = train_test_split(X,y,random_state=1)
dt.fit(X_train, y_train)
gs = GridSearchCV(dt,param_grid,scoring='f1',cv=5)
#Create a gs object, with param variable and f1 scord and 5 fold CV
gs.fit(X,y)# calculate the best possible prunign

print("Best param grid :", gs.best_params_) # print the 3 best combination of pruning methdo
print("best score :",gs.best_score_) # prints the best score for the best model
#prints the best set of method for prunign the tree


#create a file with graphviz
#dot_file = export_graphviz(dt,feature_names=feature_names)
#graph = graphviz.Source(dot_file)
#graph.render(filename ='tree',format='png',cleanup =True)
#create a file with graphviz







#model.fit(X_train,y_train)

#kf = KFold(n_splits = 5,shuffle = True)
#for criterion in ['gini','entropy']:
   # print("Decision Tree - {}".format(criterion))
   # accuracy = []
   # precision = []
    #recall = []
   # for train_index, test_index in kf.split(X):
  #      X_train, X_test = X[train_index], X[test_index]
   #     y_train, y_test = y[train_index], y[test_index]
   #     dt = DecisionTreeClassifier(criterion=criterion)
   #     dt.fit(X_train,y_train)
    #    y_pred = dt.predict(X_test)
   #     accuracy.append(accuracy_score(y_test, y_pred))
    #    precision.append(precision_score(y_test,y_pred))
    #    recall.append(recall_score(y_test,y_pred))
    #print("accuracy:",np.mean(accuracy))
   # print("precision:",np.mean(precision))
   # print("recall:",np.mean(recall))




#print("accuracy:", model.score(X_test, y_test))
#y_pred = model.predict(X_test)
#print("precision:", precision_score(y_test, y_pred))
#print("recall:", recall_score(y_test, y_pred))


