import os

def check_predict_output_score():
    path = './prediction_data'
    files = os.listdir(path)
    file = [f for f in files if 'result' in f]

    if file:
        return True
    else:
        return False