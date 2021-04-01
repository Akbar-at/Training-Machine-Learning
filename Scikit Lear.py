import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def specifity_score(y_true,y_pred): # make a function that returns the specifity score
    p,r,f,s = precision_recall_fscore_support(y_true,y_pred) #the fucntuon recision_recall_fscord [0] means r
    return r[0]
sensitivity_score = recall_score # state that the sensitivity score vlaue is the same as the recall score value
df = pd.read_csv('titanic.csv')

df['male'] = df['Sex'] == 'male'

X = df[['Fare','Pclass','male','Age','Siblings/Spouses','Parents/Children']].values # state X as an array of data that are easily processed for scikit-learn (machine friendly)

y = df['Survived'].values #the real data of survivor
X_train, X_test, y_train, y_test = train_test_split(X,y) #create a test(x,y) and test(x,y)

model = LogisticRegression() #initiate this for LogisiticRegression purposes
model.fit(X_train,y_train)#using the data to create a line that is best fit (essentially training the system for predicting
#print(model.coef_ ,model.intercept_) #this prints the line eq (coef is ax+by and intercept is c)



y_pred = model.predict(X_test)#predict the dataset (requires fitting first)
print(f"precision: {precision_score(y_test, y_pred):.5f}")

#print ((Y == y_pred).sum())#counts the amount of predictions that are correct
#print(Y.shape[0])#returning the size of the data
#print((Y == y_pred).sum()/Y.shape[0])#scoring the prediction (percentage)


#print(model.score(X_test,y_test))#model.score gives the same result as the above
#train_test_split(X, y, train_size=0.6)# changes the perimeter of the test and train ratio of the data
#print("whole dataset :",X.shape,y.shape) #rints the number of instances of the whole data
#print("training set :",X_train.shape,y_train.shape) #prints the number of instances of the data for trainign the model
#print("test set :",X_test.shape, y_test.shape)# prints the number of instances of the data used for testing the model

#print("accuracy:", accuracy_score(y_test, y_pred)) #prints the accuracy value
#print("precision:", precision_score(y_test, y_pred)) #prints precision value
#print("recall:", recall_score(y_test, y_pred))# prints recall score value
#print("f1 score:", f1_score(y_test, y_pred))# prints f1 score value
#print("sensitivity:",sensitivity_score(y_test,y_pred))# prints the sensitivity score (recall)
#print("specifity :", specifity_score(y_test, y_pred)) #prints the specify score




#fpr,tpr,thresholds = roc_curve(y_test, y_pred_proba[:,1])
#fpr1,tpr1,thresholds1 = roc_curve(y_test, y1_pred_proba[:,1])
#plt.plot(fpr,tpr)
#plt.plot(fpr1,tpr1)
#plt.plot([0,1],[0,1], linestyle="--")
#plt.xlim([0.0,1.0])
#plt.ylim([0.0,1.0])
#plt.xlabel('1-specifity')
# #plt.ylabel('sensitivity')
#plt.show()