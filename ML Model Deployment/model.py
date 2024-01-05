import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,recall_score,f1_score
from sklearn import model_selection
import matplotlib.pyplot as plt
import  seaborn as sns
import pickle
df = pd.read_csv("diabetes_data_upload.csv")
print(df.head(5))

df.info()

X=df.drop(['class'],axis=1)
y=df['class']

objlist = X.select_dtypes(include = "object").columns
print(objlist)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feat in objlist:
    X[feat] = le.fit_transform(X[feat].astype(str))
print (X.info())
print(X.head())


#data normalization
scaler=StandardScaler()
scaler.fit(X)
s_data=scaler.transform(X)
#print(s_data.head())
X=s_data
y=df['class']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=1)


#model building
classifier = svm.SVC(kernel='linear',random_state=0)
classifier.fit(X_train, y_train)

kfold=model_selection.KFold(n_splits=10)
scoring='accuracy'
acc_classifier= cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=kfold,scoring=scoring)
v=acc_classifier.mean()
print(v)

y_predict_classifier = classifier.predict(X_test)
acc=accuracy_score(y_test,y_predict_classifier)
#print(acc)
x_predict_classifier = classifier.predict(X_train)
ac=accuracy_score(y_train,x_predict_classifier)
#print(ac)

prec= precision_score(y_test,y_predict_classifier,pos_label='Negative')
#print(prec)
rec= recall_score(y_test,y_predict_classifier,pos_label='Negative')
#print(rec)
f1= f1_score(y_test,y_predict_classifier,pos_label='Negative')
#print(f1)
results= pd.DataFrame([['SVM',acc,v,prec,rec,f1]], columns= ['Model','Accuracy','Cross_val_Accuracy','Precision','Recall','F1 Score'])
print(results)


cm_classifier= confusion_matrix(y_test,y_predict_classifier)
plt.title('confusion matrix of the svm classifier')
sns.heatmap(cm_classifier,annot=True,fmt='d')
print(plt.show())

#def f_impoortances(coef, names):
   # imp=coef
    #imp,names=zip(*sorted(zip(imp,names)))
    #plt.barh(range(len(names)),imp,align='center')
    #plt.ytricks(range(len(names)),names)
    #plt.show()

#f_impoortances(svm.coef,X)


input_data = (30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
std=scaler.transform(input_data_reshaped)
prediction=classifier.predict(std)
print(prediction)
if (prediction[0] =='Negative'):
   print('non diabetic')
else:
   print('diabetic')

#pickle
pickle.dump(classifier,open("model.pkl", "wb"))
pickle.dump(scaler,open('scaler.pkl','wb'))