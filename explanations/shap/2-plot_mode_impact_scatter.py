""" Plot model output impact for top modes as a scatter plot of individual SHAP explanations """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Load processed data sets
x_train = pd.read_csv('../../data/processed/x_train.csv',index_col=0)
x_test  = pd.read_csv('../../data/processed/x_test.csv',index_col=0)
x_valid = pd.read_csv('../../data/processed/x_valid.csv',index_col=0)
y_train = pd.read_csv('../../data/processed/y_train.csv',index_col=0)
y_test  = pd.read_csv('../../data/processed/y_test.csv',index_col=0)
y_valid = pd.read_csv('../../data/processed/y_valid.csv',index_col=0)
print("Loaded processed data")

X_test = shap.sample(x_test,500)
X_test.columns = [i[1:] for i in X_test.columns]
shap_values = np.load('./data/shap_values.npy')
assert shap_values.shape[1] == X_test.shape[0]

shap.plots._labels.labels['FEATURE_VALUE_LOW'] = '$-$'
shap.plots._labels.labels['FEATURE_VALUE_HIGH'] = '$+$'
shap.summary_plot(shap_values[0], X_test,class_names=['PT1'], show=False, max_display=6,
                  color_bar_label='Displacement along mode')
plt.xlabel('Impact on model output (fs)')
plt.ylabel('Normal mode')
plt.savefig('figures/mode_impact_scatter.png',bbox_inches='tight',dpi=300)
#plt.show()
