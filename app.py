from flask import Flask, render_template, request, flash
import pickle 
import numpy as np
import pickle

with open('classification_model_pipeline','rb') as f:
    Classification_Model = pickle.load(f)

with open('regression_model_pipeline','rb') as f:
    Regression_Model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    # print(request.form)
    receive_features = [float(x) for x in request.form.values()]
    features = [np.array(receive_features)]
    # print(features[0])
    # print(features)
    pred_class = Classification_Model.predict(features)
    if pred_class[0] == 0:
        # print(f'The classification predictions is {pred_class[0]}')
        return render_template('results.html', message = 'Sorry, You are',pred_class='Defaulted',pred_roi='--', pred_ela='--', pred_emi='--')
    elif pred_class[0] == 1:
        pred_reg = Regression_Model.predict(features)
        # print(f'The classification predictions is {pred_class[0]}')
        # print(f'the Regression prediction are {pred_reg}')
        return render_template('results.html', message = 'You are', pred_class='Not Defaulted', pred_roi=round(pred_reg[0][0],2), pred_ela=round(pred_reg[0][1],2), pred_emi=round(pred_reg[0][2],2))



if __name__ == '__main__':
    app.run()