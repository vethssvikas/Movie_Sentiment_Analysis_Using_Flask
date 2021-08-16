#import libraries
import numpy as np
from flask import Flask, render_template,request
import pickle#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [x for x in request.form.values()]

    prediction = model.predict(int_features)
    output = prediction[0]
    return render_template('index.html', prediction_text='The sentiment is :{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)