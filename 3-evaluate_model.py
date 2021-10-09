""" Load ANN and evaluate it over the validation data """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
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
 
# evaluate loaded model on test data
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
loaded_model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['mse','mae'])

loaded_model.fit(x_train, y_train, validation_data=(x_valid,y_valid), epochs=0, batch_size=256)

y_predict = loaded_model.predict(x_test)

R_1 = np.corrcoef(y_test.values[:,0].astype(float),y_predict[:,0].astype(float))[1,0]
MAD = np.mean(np.abs(y_test.values-y_predict))
RMSD1 = np.sqrt(np.mean((y_test.t1-y_predict[:,0])**2))
print("R2_1 = ",R_1*R_1)
print("RMSD = ",RMSD1)
print("MAD = ",MAD)

fig, ax = plt.subplots()
plt.scatter(y_test.values[:,0],y_predict[:,0],s=32,label=('proton transfer, R$^2$: %.2f' % (R_1*R_1)))
plt.xlabel('Actual time (fs)')
plt.ylabel('Predicted time (fs)')
#plt.legend()
#plt.title('Time to proton transfer, MAE: %.1f fs, RMSD: %.1f fs' % (MAD, RMSD1))
textstr = '\n'.join((
    '{}{:.2f}{}'.format("RMSD: ",RMSD1," fs"),
    '{}{:.2f}{}'.format("MAE: ",MAD," fs"),
    '{}{:.2f}{}'.format("R$^2$: ",R_1*R_1, "   ")))

# these are matplotlib.patch.Patch properties
props = dict(facecolor='white', edgecolor='white',alpha=0.0)

# place a text box in upper left in axes coords
ax.text(0.65, 0.35, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.plot(np.arange(0,310),np.arange(0,310),color='black',lw=4,ls='--')
plt.ylim([0,310])
plt.xlim([0,310])
plt.savefig('figures/proton_transfer_regression.png',bbox_inches='tight',dpi=300)
#plt.show()
