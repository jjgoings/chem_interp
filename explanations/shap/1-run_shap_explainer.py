""" Run SHAP explanations on the trained proton transfer model """
import numpy as np
import pandas as pd
import shap
from tensorflow.keras.optimizers import Adam
from keras.models import model_from_json

MODEL = '../../model/proton_transfer_model.json'
MODEL_WEIGHTS = '../../model/proton_transfer_model_weights.h5'

# Load processed data sets
x_train = pd.read_csv('../../data/processed/x_train.csv',index_col=0)
x_test  = pd.read_csv('../../data/processed/x_test.csv',index_col=0)
x_valid = pd.read_csv('../../data/processed/x_valid.csv',index_col=0)
y_train = pd.read_csv('../../data/processed/y_train.csv',index_col=0)
y_test  = pd.read_csv('../../data/processed/y_test.csv',index_col=0)
y_valid = pd.read_csv('../../data/processed/y_valid.csv',index_col=0)
print("Loaded processed data")

with open(MODEL, 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(MODEL_WEIGHTS)
print("Loaded model from disk")
 
# evaluate loaded model on test data
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
loaded_model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['mse','mae'])

loaded_model.fit(x_train, y_train, validation_data=(x_valid,y_valid), epochs=0, batch_size=64)

X_train = shap.sample(x_train,500)
X_test = shap.sample(x_test,500)

print('Doing explainer...')
explainer = shap.KernelExplainer(loaded_model.predict, X_train)
print('Doing SHAP values...')
shap_values = explainer.shap_values(X_test,l1_reg = "num_features(144)")
np.save('./data/shap_values.npy',np.asarray(shap_values),allow_pickle=False)
