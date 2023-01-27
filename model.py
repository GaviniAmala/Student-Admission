import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
df=pd.read_csv(r"C:\Users\Admin\Downloads\binary.csv")
print(df.head())
from sklearn.ensemble import ExtraTreesClassifier
model =ExtraTreesClassifier()
x=df[['gre', 'gpa', 'rank']].to_numpy()
y=df.admit
model.fit(x,y)
print(model.score(x,y))
print(model.predict_proba(x))
print(model.predict([[789,3.43,1]]))
#make pickle file of our model
pickle.dump(model, open("model.pkl", "wb"))
model=pickle.load(open('model.pkl','rb'))