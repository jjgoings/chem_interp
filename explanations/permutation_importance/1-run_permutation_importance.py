""" Perform the permutation importance tests on the trained proton transfer model """
import pandas as pd
from keras.models import model_from_json
from keras.wrappers.scikit_learn import KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance
from eli5.formatters.as_dataframe import format_as_dataframe

RANDOM_SEED = 200
PERM_ITER = 200  # number of iterations in permutation testing
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

def base_model():
    """ Load and return pre-trained model """
    with open(MODEL, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights(MODEL_WEIGHTS)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mae'])
    return model

model = KerasRegressor(build_fn=base_model, epochs=0)
model.fit(x_train, y_train) # doesn't actually fit since epochs=0, but needed to load model properly

# There is some debate whether you should use train or valid for explaining ... we use test data
# but results don't really change if you use valid ... feel free to play around
perm = PermutationImportance(model, random_state=RANDOM_SEED, n_iter=PERM_ITER).fit(x_test,y_test)
explanation = eli5.explain_weights(perm, feature_names = x_test.columns.tolist())
exp = format_as_dataframe(explanation)
exp.to_csv('./data/permutation_importance_results.csv')
