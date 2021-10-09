""" Top routine to set up feed forward ANN on proton transfer reactions """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import save_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam

NUM_NEURONS = 768
DROPOUT = 0.25

# where to save model after preparation
MODEL = './model/proton_transfer_model.json'
MODEL_WEIGHTS = './model/proton_transfer_model_weights.h5'

# Load processed data sets
x_train = pd.read_csv('data/processed/x_train.csv',index_col=0)
x_test  = pd.read_csv('data/processed/x_test.csv',index_col=0)
x_valid = pd.read_csv('data/processed/x_valid.csv',index_col=0)
y_train = pd.read_csv('data/processed/y_train.csv',index_col=0)
y_test  = pd.read_csv('data/processed/y_test.csv',index_col=0)
y_valid = pd.read_csv('data/processed/y_valid.csv',index_col=0)

model = Sequential()
model.add(Dropout(0.0,input_shape=(144,)))
model.add(Dense(NUM_NEURONS))
model.add(LeakyReLU())
model.add(Dropout(DROPOUT))
model.add(Dense(NUM_NEURONS))
model.add(LeakyReLU())
model.add(Dropout(DROPOUT))
model.add(Dense(NUM_NEURONS))
model.add(LeakyReLU())
model.add(Dropout(DROPOUT))
model.add(Dense(1,activation='linear'))

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['mse','mae'])

model.fit(x_train, y_train, validation_data=(x_valid,y_valid), epochs=50, batch_size=64)

y_predict = model.predict(x_test)

save_model(model, MODEL, MODEL_WEIGHTS)
print("Saved model to disk")

plt.scatter(y_test.values[:,0],y_predict[:,0])
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.plot(np.arange(0,310),np.arange(0,310),color='black',lw=4,ls='--')
plt.show()
