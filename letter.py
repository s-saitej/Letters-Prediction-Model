import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('letterdata.csv')
print(data)
x = data.iloc[:,1:]
y = data.iloc[:,:1]
print(x.isnull().any())
print(y.isnull().any())
from sklearn import model_selection
train_data,test_data,train_target,test_target = model_selection.train_test_split(x,y)
print(train_data)
print(test_data)
from sklearn import svm


classification = svm.SVC(kernel = 'poly', C=1e2, gamma = 10)
fitting = classification.fit(train_data,train_target)
result = classification.predict(test_data)


plt.hist(test_target, color = 'blue')
plt.hist(result, color = 'orange')
from sklearn import metrics
acuuracy = metrics.accuracy_score(result,test_target)
