from flask import Flask, render_template, request
import pickle
import requests
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = pickle.load(open('../emp_attrition.pickle', 'rb'))


@app.route('/')
def hello_world():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
