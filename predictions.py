import pandas as pd
df = pd.read_csv('Anne_DNA_66.csv')
df = df[df['Class'] != 'interphase']

df_x = pd.read_csv('Anne_DNA_66.csv')
trimmed_interphase = df_x[df_x['Class'] == 'interphase'].head(150)
df = pd.concat([df, trimmed_interphase],axis=0) # Add interphase but balanced

y = df['Class'].values
df = df.drop('TableNumber', 1)
df = df.drop('ImageNumber', 1)
df = df.drop('ObjectNumber', 1)
df = df.drop('Class', 1)
df = df.drop('Nuclei_AreaShape_EulerNumber', 1)

df_norm = (df - df.mean()) / (df.max() - df.min())
df_norm.values.shape

# Convert y into Y
nb_classes = 23
import numpy as np
from keras.utils import np_utils, generic_utils

# Convert labels to numeric
y_unique = np.unique(y)
dic = {}

for i, label in enumerate(y_unique):
    dic[label] = i
print dic

y_numeric = []
for el in y:
    y_numeric += [dic[el]]
    
y_numeric # now a 2000 label vector
Y = np_utils.to_categorical(y_numeric, nb_classes)

print Y.shape

Y_train = Y

X_train = df_norm.values

from sklearn.cross_validation import StratifiedKFold

#print y_numeric
skl = StratifiedKFold(y_numeric, n_folds=5)

y_numeric = np.array(y_numeric)

for train,test in skl:
    print len(train), len(test)
    
X_4 = X_train[train]
Y_4 = Y[train]
y_numeric_4 = y_numeric[train]

X_eval = X_train[test]
Y_eval = Y[test]
y_numeric_eval = y_numeric[test]

X_stratified = np.append(X_4,X_eval,axis=0)
Y_stratified = np.append(Y_4,Y_eval,axis=0)

X_stratified.shape
Y_stratified.shape

df_full = pd.read_csv('../objects.csv')

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(X_train,y_numeric)

training_values = df.columns.values
df_full_trimmed = df_full[training_values]

df_full_norm = (df_full_trimmed - df_full_trimmed.mean()) / (df_full_trimmed.max() - df_full_trimmed.min())
df_full_norm.values.shape

predictions = rf.predict(df_full_norm.values)

print predictions
