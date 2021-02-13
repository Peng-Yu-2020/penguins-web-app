import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

penguins = pd.read_csv('penguins_cleaned.csv')

df = penguins.copy()

target = 'species'
encode = ['sex', 'island']

for col in encode:
    
    dumy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dumy], axis = 1)
    del(df[col])
    
target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}

def target_map(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_map)

X = df.drop('species', axis=1)
y = df['species']

clf = RandomForestClassifier()
clf.fit(X, y)

pickle.dump(clf, open('penguins_clf.pkl', 'wb'))
    