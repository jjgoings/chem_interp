""" Given a model, keep training on the proton transfer data """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

MODEL = './model/proton_transfer_model.json'
MODEL_WEIGHTS = './model/proton_transfer_model_weights.h5'
loaded_model = load_model(MODEL, MODEL_WEIGHTS)
print("Loaded model from disk")

# Load processed data sets
x_train = pd.read_csv('data/processed/x_train.csv',index_col=0)
x_test  = pd.read_csv('data/processed/x_test.csv',index_col=0)
x_valid = pd.read_csv('data/processed/x_valid.csv',index_col=0)
y_train = pd.read_csv('data/processed/y_train.csv',index_col=0)
y_test  = pd.read_csv('data/processed/y_test.csv',index_col=0)
y_valid = pd.read_csv('data/processed/y_valid.csv',index_col=0)
print("Loaded processed data")

# evaluate loaded model on test data
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
loaded_model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['mse','mae'])

checkpoint = ModelCheckpoint(MODEL_WEIGHTS,save_freq=50)
callbacks_list = [checkpoint]

loaded_model.fit(x_train, y_train, validation_data=(x_valid,y_valid), 
                 epochs=50, batch_size=64,callbacks=callbacks_list,verbose=2)

y_predict = loaded_model.predict(x_test)

save_model(loaded_model, MODEL, MODEL_WEIGHTS)
print("Saved model to disk")

plt.scatter(y_test.values[:,0],y_predict[:,0])
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.plot(np.arange(0,310),np.arange(0,310),color='black',lw=4,ls='--')
plt.show()
