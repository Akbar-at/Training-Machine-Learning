from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
X,y = load_digits(return_X_y=True)


print(X.shape,y.shape)

#this bundles of code :
#ngetes salahnya model neuron dimana dengan ngelihat gambarnya dulu pertama terus ngeprin true number sama predicted number
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)
mlp = MLPClassifier(random_state=2)
mlp.fit(X_train,y_train)

print(mlp.score(X_test,y_test))
y_pred = mlp.predict(X_test)
incorrect = X_test[y_pred != y_test] #displays the incorrect
incorrect_true = y_test[y_pred != y_test] #the true value
incorrect_pred = y_pred[y_pred != y_test] #pred value

j = 5 #5 artinya false pred ke 5 
plt.matshow(incorrect[j].reshape(8,8),cmap=plt.cm.gray)
plt.xticks(())    # matshow nunjukin gambar nya , reshape(8,8) karena gambarnya 8 klai 8 pixel
plt.yticks(())
plt.show()
print("true value:",incorrect_true[j])
print("predicted value:",incorrect_pred[j])




'''
plt.matshow(x.reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.show()
print(mlp.predict([X_test[6]]))

print("accuracy:", mlp.score(X_test, y_test))
x=X_test[6]
'''

'''
plt.scatter(X[y==0][:,0],X[y==0][:,1],s = 100, edgecolors='k',marker='*')
plt.scatter(X[y==1][:,0],X[y==1][:,1],s = 50 , edgecolors='k',)

plt.show()
'''