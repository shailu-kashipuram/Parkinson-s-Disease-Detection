import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

import warnings
warnings.filterwarnings('ignore')



app = Flask(__name__)




@app.route("/about")
def about1():
    return render_template("about.html")


@app.route('/')
@app.route('/home')
def home1():
	return render_template('home.html')



@app.route('/predict',methods=['POST'])
def predict():
    int_features= [float(x) for x in request.form.values()]
    print(int_features,len(int_features))
    final4=[np.array(int_features)]
    model = joblib.load('model.sav')
    predict = model.predict(final4)

    if predict == 0:
        output = "NEGATIVE, PATIENT IS NOT SUFFER FROM PARKINSON's DISEASE!" 
    elif predict == 1:
        output = "POSITIVE, PATIENT IS SUFFER FROM PARKINSON's DISEASE!"  
    
    
    return render_template('prediction.html', output=output)



if __name__ == "__main__":
    app.run(debug=False)
