from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    gre = float(request.form['gre'])
    gpa = float(request.form['gpa'])
    rank = int(request.form['rank'])
    features = np.array([gre, gpa, rank]).reshape(1, -1)
    probability = model.predict_proba(features)[0][1]
    admit=model.predict([[gre,gpa,rank]])
    if admit[0]==0:
         return render_template('predict.html', admit="can not be admitted")
    else:
        return render_template('predict.html',admit="can be admitted")



if __name__ == '__main__':
    app.run()