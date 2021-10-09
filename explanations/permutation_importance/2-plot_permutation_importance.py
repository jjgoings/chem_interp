""" Plot the top features identified by permutation importance testing of proton transfer """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

NUM_FEATURES_TO_PLOT = 6  # number of top features to include in plot

df = pd.read_csv('./data/permutation_importance_results.csv')

feature = df['feature']
weight  = df['weight']
err     = df['std']

fig, ax = plt.subplots()

features = [i[1:] for i in feature[:NUM_FEATURES_TO_PLOT]]
y_pos = np.arange(len(feature[:NUM_FEATURES_TO_PLOT]))
ax.barh(y_pos,weight[:NUM_FEATURES_TO_PLOT],xerr=err[:NUM_FEATURES_TO_PLOT],lw=2)
ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.invert_yaxis()
ax.grid(False)
plt.ylabel('Normal mode')
plt.xlabel('Mean Increase in Error (MAE, fs)')
plt.savefig('figures/mode_importance.png', dpi=600)
#plt.show()
