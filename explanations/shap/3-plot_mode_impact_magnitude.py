""" Plot the average magnitude of top mode's impact on model output """
import numpy as np
import matplotlib.pyplot as plt

shap_values = np.load('./data/shap_values.npy')

MAX_DISPLAY = 6
shap_values = shap_values[0]
feature_order = np.argsort(np.sum(-np.abs(shap_values), axis=0))
feature_inds = feature_order[:MAX_DISPLAY]

fig, ax = plt.subplots()

feature_names = np.arange(1,shap_values.shape[1]+1)
y_pos = np.arange(len(feature_inds))
global_shap_values = np.abs(shap_values).mean(0)
plt.barh(y_pos, global_shap_values[feature_inds],color='g')
plt.gca().set_yticklabels([feature_names[i] for i in feature_inds])

ax.set_yticks(y_pos)
ax.invert_yaxis()
ax.grid(False)
plt.ylabel('Normal mode')
plt.xlabel('Average impact on model output magnitude (fs)')
plt.savefig('figures/mode_impact_magnitude.png',bbox_inches='tight',dpi=600)
#plt.show()
