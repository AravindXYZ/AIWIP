import pandas as pd
dataset = pd.read_csv("Datasets/insurance_pre.csv")
#print(dataset)
dummies = pd.get_dummies(dataset,drop_first=True)
dummies = dummies.astype(int)
#print(dummies)
independent=dummies[['age', 'bmi', 'children','sex_male', 'smoker_yes']]
dependent=dataset['charges']
#print(type(dummies))
print(dependent)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(independent,dependent,test_size=0.30,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

weight=regressor.coef_
print('weight',weight)

bais=regressor.intercept_
print('bais',bais)

y_pred=regressor.predict(X_test)
#print('y_pred',y_pred)

from sklearn.metrics import r2_score
r_score=r2_score(y_test,y_pred)

print('r_score',r_score)

import pickle
filePath="Deployed/MLR.sav"
pickle.dump(regressor,open(filePath,'wb'))

loaded_model=pickle.load(open(filePath,'rb'))
data = [[30, 25, 2, 1, 0],[45, 30, 0, 0, 1]]#
df = pd.DataFrame(data,columns=['age', 'bmi', 'children', 'sex_male', 'smoker_yes'])
result=loaded_model.predict(df)#or passin #X_test

for d in range(len(data)):
    print("-------------------")
    for i in range(df.columns.size):
        print(df.columns[i]+" = ",data[d][i])
    print("cost = ",result[d])
print("-------------------")
print('r_score2',r2_score([4500,29000],result))