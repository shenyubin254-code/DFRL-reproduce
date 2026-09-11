import os

from flask import Flask, request

from drd2_svm import predict_drd2

app = Flask(__name__)

PORT = int(os.environ.get("DRD2_SERVER_PORT", "5002"))


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()['SMILES_LIST']
    value = predict_drd2(data)[0]
    return {'result': str(value)}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
