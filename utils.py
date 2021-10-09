""" Utilities to help manipulate models """
from tensorflow.keras.models import model_from_json

def load_model(model_json_file, model_weights_file):
    with open(model_json_file, 'r') as model_architecture:
        model = model_from_json(model_architecture.read())
    # load weights into new model
    model.load_weights(model_weights_file)
    return model

def save_model(model, model_json_file, model_weights_file):
    model_json = model.to_json()
    with open(model_json_file, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_weights_file)
