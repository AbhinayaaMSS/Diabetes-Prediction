import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#create flask app
app=Flask(__name__)

model= pickle.load(open("model.pkl", "rb"))
sc=pickle.load(open('scaler.pkl','rb'))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route('/taketest')
def taketest():
    return render_template('taketest.html')

@app.route("/predict", methods=["POST","GET"])
def predict():
    age = request.form['age']
    gender= request.form['gender']
    poly = request.form['poly']
    polyd = request.form['polyd']
    weight = request.form['weight']
    weak = request.form['weak']
    polyp = request.form['polyp']
    gene = request.form['gene']
    vis = request.form['vis']
    itch = request.form['itch']
    irri = request.form['irri']
    delay = request.form['delay']
    partial = request.form['partial']
    muscle = request.form['muscle']
    alop = request.form['alop']
    obese = request.form['obese']


    features = np.array([(age, gender, poly, polyd, weight, weak, polyp, gene, vis, itch, irri, delay, partial, muscle, alop, obese)])
    #input_data_as_numpy_array = np.asarray(features)
    #input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction=model.predict(sc.transform(features))
    return render_template("taketest.html",prediction_text='The patient has diabetes : {}'.format(prediction))

if __name__ == "__main__":
   app.run(debug=True)


