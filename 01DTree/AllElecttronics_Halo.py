import pandas as pd
from sklearn import tree
from sklearn import preprocessing

df = pd.read_csv('D:/Wheat-College/01DTree/AllElectronics.csv', encoding='utf-8',index_col='RID')

# transform dummy variables
df = pd.get_dummies(df,columns=['age', 'income','student','credit_rating','class_buys_computer'])

# spilt x_train, y_train
y_train = df['class_buys_computer_yes']
x_train = df.drop(['class_buys_computer_no','class_buys_computer_yes'],axis=1)


# Using decisicon tree for classficiation

dtr = tree.DecisionTreeClassifier(criterion='entropy')
dtr.fit(x_train,y_train)
print("clf: " + str(dtr))


# Visualize model
with open("allElectronicInformationGainOri_Halo.dot",'w') as f:
    f = tree.export_graphviz(dtr,out_file=f )

# constract test data
x_test = x_train.iloc[1:2,:]

x_test.loc[2,'age_middle_aged'] = 1
x_test.loc[2,'age_youth'] = 0

# predict
y_test = dtr.predict(x_test)

print(y_test)



