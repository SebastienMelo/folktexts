#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from folktexts.acs import ACSDataset
from sklearn.tree import DecisionTreeRegressor
import glest
from folktexts.acs import ACSTaskMetadata
from deferral_experiment.regret_helpers import compute_regret_CL, get_constant_utilty, get_threshold_from_utility
#%%

llama1 = pd.read_csv('deferral_experiment/merged_llama1_results.csv')
llama70 = pd.read_csv('deferral_experiment/merged_llama70_results.csv')
# %%

X = llama1.drop(columns=['risk_score', 'label']).values
y = llama1['label'].values
S = llama1['risk_score'].values


S_llama70 = llama70['risk_score'].values

#%%


t_target = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99]
U = get_constant_utilty(100, t_target)  # (n_utilities, 2, 2)
t = get_threshold_from_utility(U) 

calibrated_classifier = LogisticRegression()
X_train, X_test, y_train, y_test, S_train, S_test, S_llama70_train, S_llama70_test = train_test_split(
    X, y, S, S_llama70, test_size=0.5, random_state=0
)

X_train, X_cal, y_train, y_cal, S_train, S_cal = train_test_split(
    X_train, y_train, S_train, test_size=max(int(len(X_train) * 0.2),4000), random_state=0
)

calibrated_classifier.fit(S_cal.reshape(-1,1), y_cal)

c_hat_train = calibrated_classifier.predict_proba(S_train.reshape(-1,1))[:, 1]
c_hat_test = calibrated_classifier.predict_proba(S_test.reshape(-1,1))[:, 1]

residuals_train = y_train - c_hat_train
residuals_test = y_test - c_hat_test
dt = DecisionTreeRegressor(max_depth = 10, min_samples_leaf= 10)
dt.fit(X_train, residuals_train)
leaf_ids = dt.apply(X_test)


gle = glest.core.GLEstimatorResiduals(None, None)
gle.fit(X_test, y_test, y_scores_cal = c_hat_test, partition = leaf_ids)
# fig = gle.plot(fig_kw=dict(figsize=(2.5, 2.5)))

r_hat = gle.honest_tree_pred

# C, H = glest_calibration_curve(
#     gle.frac_pos_, gle.counts_, gle.mean_scores_, remove_empty=False
# )
t=0.5
# bins = gle.partitioner.bins_
# binids = np.searchsorted(bins[1:-1], S)

a = c_hat_test[:, None] >= t
RCL = compute_regret_CL(c_hat_test, t, a)  # (n, k)
# RCL = RCL.mean(axis=0)

RGL = compute_regret_CL(c_hat_test + r_hat, t, a)  # (n, k)
# RGL = RGL.mean(axis=0)


#%%
RGL.shape
# %%
# Get the indices of the 10% samples with the biggest regret
n_samples = len(RGL)
top_10_percent = int(0.1 * n_samples)
top_regret_indices = np.argsort(RGL.flatten())[-top_10_percent:]
top_regret_samples = RGL.flatten()[top_regret_indices]
# %%
top_regret_indices
# %%
# Create a mask for the top regret samples
defer_mask = np.zeros(len(S_test), dtype=bool)
defer_mask[top_regret_indices] = True

# Get llama70 predictions for the test set (assuming same order as llama1)
llama70_test = llama70.iloc[len(S_train):len(S_train)+len(S_test)]
llama70_predictions = llama70_test['risk_score'].values

# Create mixed predictions: use llama70 for high regret samples, llama1 otherwise
mixed_predictions = S_test.copy()
mixed_predictions[defer_mask] = llama70_predictions[defer_mask]