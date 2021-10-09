""" Plots SHAP value (impact) as a function of mode displacement """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
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

# need to obtain the scaler so we can convert back to "real"  molecular displacements ...
# so load up the raw data once more
scal = StandardScaler()
x_train_raw = pd.read_csv('../../data/split_raw/x_train.csv',index_col=0)
scal.fit(x_train_raw.values)

# load up the model
with open(MODEL, 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(MODEL_WEIGHTS)
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
loaded_model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['mse','mae'])
loaded_model.fit(x_train, y_train, epochs=0, batch_size=128)
print("Loaded model from disk")

X_test = shap.sample(x_test,500)
shap_values = np.load('./data/shap_values.npy')
assert shap_values.shape[1] == X_test.shape[0]
print("Loaded SHAP values")

X_train = shap.sample(x_train,500)
explainer = shap.KernelExplainer(loaded_model.predict, X_train)
X_test = shap.sample(x_test,500)

# Convert back to chemical, unscaled, units, i.e. sqrt(amu) * Angstrom
X_test_features = scal.inverse_transform(X_test.values)*np.sqrt(5.4857990943e-4)*0.529177211
X_test = pd.DataFrame(X_test_features, index=X_test.index, columns=X_test.columns)

# we want to look more closely at modes 119, 24, and 5 from earlier results
for mode in [119,24,5]:
    shap.dependence_plot("q"+str(mode), shap_values[0], X_test, interaction_index="q"+str(mode),show=False)
    plt.xlabel('Mode '+str(mode)+' displacement ($\sqrt{\mathrm{amu}} \cdot \mathrm{\AA}$)')
    plt.ylabel('SHAP value (fs)')
    plt.savefig('figures/mode_q'+str(mode)+'_dependence.png',bbox_inches='tight',dpi=300)
    plt.close()


