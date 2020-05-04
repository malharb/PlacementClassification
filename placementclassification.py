import pandas as pd
df = pd.read_csv('placement.csv')

#making dummy variables
gender_dummy = pd.get_dummies(df['gender'],drop_first = True)
ssc_b_dummy = pd.get_dummies(df['ssc_b'],drop_first = True)
hsc_b_dummy = pd.get_dummies(df['hsc_b'],drop_first = True)
hsc_s_dummy = pd.get_dummies(df['hsc_s'],drop_first = True)
degree_t_dummy = pd.get_dummies(df['degree_t'],drop_first = True)
workex_dummy = pd.get_dummies(df['workex'],drop_first = True)
specialisation_dummy = pd.get_dummies(df['specialisation'],drop_first = True)

df = df.drop(['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation'],axis = 1)
df = pd.concat([df,gender_dummy,ssc_b_dummy,hsc_b_dummy,degree_t_dummy,workex_dummy,specialisation_dummy],axis=1)
del df['sl_no']

#Preprocessing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
toscale = df[['ssc_p','hsc_p','degree_p','etest_p','mba_p']]
scaler.fit(toscale)
scaled_df = pd.DataFrame(scaler.transform(toscale),columns=['ssc_p','hsc_p','degree_p','etest_p','mba_p'])
scaled_df = pd.concat([scaled_df,gender_dummy,ssc_b_dummy,hsc_b_dummy,degree_t_dummy,workex_dummy,specialisation_dummy],axis=1)

salary = df['salary']
del df['salary']

#Splitting the data
from sklearn.model_selection import train_test_split
x = df.drop('status',axis=1)
y = df['status']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)

#Using logistic regression
from sklearn.linear_model import LogisticRegression
logreg_model = LogisticRegression()
logreg_model.fit(x_train,y_train)
logreg_predictions = logreg_model.predict(x_test)

#Using a random forest 
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 500)
rfc.fit(x_train,y_train)
rfc_predictions = rfc.predict(x_test)

#Using KNN
x2 = scaled_df
y2 = pd.get_dummies(df['status'],drop_first = True)
x2_train,x2_test,y2_train,y2_test = train_test_split(x2,y2,test_size = 0.25)

from sklearn.neighbors import KNeighborsClassifier
error_value = []
totalerror = 0
for i in range(1,51):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x2_train,y2_train)
    knn_predictions = knn.predict(x2_test)
    knn_predictions = pd.Series(knn_predictions)
    for x in range(1,len(knn_predictions)):
        totalerror = totalerror + (list(knn_predictions) != list(y2_test['Placed']))
    meanerror = totalerror/len(knn_predictions)
    error_value.append(meanerror)

errorvaluesdf = pd.DataFrame(error_value,index=range(1,51),columns=['EV'])
#print(errorvaluesdf[errorvaluesdf['EV']==errorvaluesdf['EV'].min()])
#KNN best with K = 1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x2_train,y2_train)
knn_predictions = knn.predict(x2_test)

#Metrics
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,logreg_predictions))
print(confusion_matrix(y_test,rfc_predictions))
print(confusion_matrix(y2_test,knn_predictions))


#
