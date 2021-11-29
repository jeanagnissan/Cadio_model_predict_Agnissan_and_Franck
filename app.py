from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction/', methods=['get', 'post'])
def my_form_post():
    data = {
        "AGE": input(request.form['age']),
        "SEXE": input(request.form['sexe']),
        "TDT": input(request.form['tdt']),
        "PAR": input(request.form['par']),
        "CHOLESTEROL": input(request.form['cholesterol']),
        "GAJ": input(request.form['gaj']),
        "ECG": input(request.form['ecg']),
        "FCMAX": input(request.form['fcmax']),
        "ANGINE": input(request.form['angine']),
        "DEPRESSION ": input(request.form['depression']),
        "PENTE": input(request.form['pente']),
    }
    #app.logger.debug(f"okokok {data}")

    dataf = pd.DataFrame(data, index=[0])
    coeur = pd.read_excel('Coeur.xlsx')

    #Normalisation des variables quantitatives
    for col in coeur.drop(['CŒUR'], axis=1).select_dtypes(np.number).columns:
        dataf[col] = dataf[col] / coeur[col].max()

    #Encodage des variables qualitatives
    for col in coeur.drop(['CŒUR'], axis=1).select_dtypes('object').columns:
        dataf[col] = dataf[col].astype('category').cat.codes

    #importer le model
    model = pickle.load(open("model.pkl", "rb"))
    result = model.predict(dataf)

    return render_template('home.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
