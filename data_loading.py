#%%
from folktexts.acs import ACSDataset

from folktexts.acs import ACSTaskMetadata
import pandas as pd
import glest 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from glest.plot import grouping_diagram_residuals
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from scipy.interpolate import griddata
import glob
import re
# %%

DATA_DIR = "notebooks/data"
TASK_NAME = "ACSIncome"
task = ACSTaskMetadata.get_task(TASK_NAME, use_numeric_qa=False)

dataset = ACSDataset.make_from_task(task=task, cache_dir=DATA_DIR)
# %%
dataset.data
# %%

df = pd.read_csv("folktexts-results/folktexts-results/model-Mixtral-8x7B-v0.1_task-ACSIncome/Mixtral-8x7B-v0.1_bench-838239935/ACSIncome_full_seed-42_hash-1998608642.test_predictions.csv")

# %%
features = dataset.get_features_data()
# %%
# Create a mapping between features and df indexes
matched_features = features.loc[df["Unnamed: 0"]].copy()

# %%
test = df.set_index(df["Unnamed: 0"].values)
# %%
test.drop(columns=["Unnamed: 0"], inplace=True)
# %%
merged_df = pd.concat([matched_features, test], axis=1)
# %%
merged_df
# %%
f = merged_df["risk_score"]
y = merged_df["label"]

merged_df.drop(columns=["risk_score", "label"], inplace=True)


#%%
# Plot mapping 

# merged_df.drop(columns=["COW", "RAC1P", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX"], inplace=True)


# ACSIncome_categories = {
#     "RAC1P": {
#         1.0: "White alone",
#         2.0: "Black or African American alone",
#         3.0: "American Indian alone",
#         4.0: "Alaska Native alone",
#         5.0: (
#             "American Indian and Alaska Native tribes specified;"
#             "or American Indian or Alaska Native,"
#             "not specified and no other"
#         ),
#         6.0: "Asian alone",
#         7.0: "Native Hawaiian and Other Pacific Islander alone",
#         8.0: "Some Other Race alone",
#         9.0: "Two or More Races",
#     }
# }
# for col, mapping in ACSIncome_categories.items():
#     if col in merged_df.columns:
#         merged_df[col] = merged_df[col].map(mapping).fillna("Unknown")
#         merged_df[col] = merged_df[col].astype("category")

#%%

merged_df


# %%
calibrated_classifier = LogisticRegression()
X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(
    merged_df.values, y.values, f.values, test_size=0.5, random_state=0
)

X_train, X_cal, y_train, y_cal, S_train, S_cal = train_test_split(
    X_train, y_train, S_train, test_size=max(int(len(X_train) * 0.2),4000), random_state=0
)

calibrated_classifier.fit(S_cal.reshape(-1,1), y_cal)

c_hat_train = calibrated_classifier.predict_proba(S_train.reshape(-1,1))[:, 1]
c_hat_test = calibrated_classifier.predict_proba(S_test.reshape(-1,1))[:, 1]

residuals_train = y_train - c_hat_train
residuals_test = y_test - c_hat_test
dt = DecisionTreeRegressor(max_depth = 7, min_samples_leaf= 10)
dt.fit(X_train, residuals_train)
leaf_ids = dt.apply(X_test)


gle = glest.core.GLEstimatorResiduals(None, None)
gle.fit(X_test, y_test, y_scores_cal = c_hat_test, partition = leaf_ids)
# %%
grouping_diagram_residuals(c_hat_test, gle.honest_tree_pred, gle.n_per_leaf, S_test,y_cal, f_cal = S_cal,leaf_ids=leaf_ids)

# %%
r_j = gle.get_rj()
# print("rj shape",r_j.shape)
partition = leaf_ids
# print("partition shape",leaf_ids.shape)
path = "folktexts-results/folktexts-results/model-Mixtral-8x7B-v0.1_task-ACSIncome/Mixtral-8x7B-v0.1_bench-838239935/"
fig_path = f"{path}figures/"
if not os.path.exists(fig_path):
    os.makedirs(fig_path)


values_honest_tree = np.zeros_like(partition, dtype=np.float64)
print(np.max(partition))
for i, leaf in enumerate(np.unique(partition)):
    mask = (partition == leaf)
    values_honest_tree[mask] = np.clip(r_j[i], a_min=0, a_max=None)
    # print(values_honest_tree[mask])
    

# values_honest_tree = r_j[partition]

print("Honest tree values",values_honest_tree)
# print("X_train shape",X_train.shape)    
plt.figure(figsize=(6, 2.8))

# Create a scatter plot instead of a contour plot
# Assuming X_test is 2D feature space
if X_test.shape[1] >= 2:
    # Create a scatter plot with color mapping
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=values_honest_tree, cmap='viridis', s=20, norm=mcolors.LogNorm(clip=True))
    plt.colorbar(scatter, label='Local GL values')
    
    plt.xlabel('Age')
    plt.ylabel('School level')
            
else:
    plt.scatter(np.arange(len(values_honest_tree)), values_honest_tree)
    plt.xlabel('Index')
    plt.ylabel('Residual values')
plt.title('Mapping of the grouping loss')
plt.tight_layout()
# Save the figure
plt.savefig(f"{fig_path}Mixtral8x7B_base_mapping_of_gl.png")
plt.close()
# %%


def process_test_prediction_file(file_path, features):
        """
        Process a single test prediction CSV file and compute GLE.
        
        Args:
            file_path (str): Path to the test prediction CSV file
            features (pd.DataFrame): Features dataset
        
        Returns:
            tuple: (gle_estimator, file_path) or (None, file_path) if error
        """
    
        # Read the CSV file
        df_current = pd.read_csv(file_path)
        
        # Create matched features for current file
        matched_features_current = features.loc[df_current["Unnamed: 0"]].copy()
        
        # Set index and drop unnamed column
        test_current = df_current.set_index(df_current["Unnamed: 0"].values)
        test_current.drop(columns=["Unnamed: 0"], inplace=True)
        
        # Merge features with predictions
        merged_df_current = pd.concat([matched_features_current, test_current], axis=1)
        
        # Extract risk scores and labels
        f_current = merged_df_current["risk_score"]
        y_current = merged_df_current["label"]
        merged_df_current.drop(columns=["risk_score", "label"], inplace=True)
        
        # Split data
        X_train_curr, X_test_curr, y_train_curr, y_test_curr, S_train_curr, S_test_curr = train_test_split(
            merged_df_current.values, y_current.values, f_current.values, test_size=0.5, random_state=0
        )
        
        X_train_curr, X_cal_curr, y_train_curr, y_cal_curr, S_train_curr, S_cal_curr = train_test_split(
            X_train_curr, y_train_curr, S_train_curr, test_size=max(int(len(X_train_curr) * 0.2), 4000), random_state=0
        )
        
        # Fit calibrated classifier
        calibrated_classifier_curr = LogisticRegression()
        calibrated_classifier_curr.fit(S_cal_curr.reshape(-1,1), y_cal_curr)
        
        c_hat_train_curr = calibrated_classifier_curr.predict_proba(S_train_curr.reshape(-1,1))[:, 1]
        c_hat_test_curr = calibrated_classifier_curr.predict_proba(S_test_curr.reshape(-1,1))[:, 1]
        
        # Calculate residuals and fit decision tree
        residuals_train_curr = y_train_curr - c_hat_train_curr
        dt_curr = DecisionTreeRegressor(max_depth=7, min_samples_leaf=10)
        dt_curr.fit(X_train_curr, residuals_train_curr)
        leaf_ids_curr = dt_curr.apply(X_test_curr)
        
        # Fit GLE
        gle_curr = glest.core.GLEstimatorResiduals(None, None)
        gle_curr.fit(X_test_curr, y_test_curr, y_scores_cal=c_hat_test_curr, partition=leaf_ids_curr)
        
        return gle_curr, file_path


# Search for CSV files containing "test_predictions" in the folktexts-results folder
test_prediction_files = glob.glob("folktexts-results/**/*test_predictions*.csv", recursive=True)

# Process each test prediction file
for file in test_prediction_files:
    print(f"\nProcessing file: {file}")
    
    
    
    
            


    # Process each test prediction file using the function
    gle_results = []
    for file in test_prediction_files:
        print(f"\nProcessing file: {file}")
        gle_result, processed_file = process_test_prediction_file(file, features)
        if gle_result is not None:
            gle_results.append((gle_result, processed_file))
            print(f"GLE fitted successfully for {processed_file}")

        print(f"\nSuccessfully processed {len(gle_results)} files")
print("Found test prediction files:")
for file in test_prediction_files:
    print(file)
# %%
gle_results
# %%
gle_results[0][0].metrics()["GL"]
# %%
# Extract GL values and model names from gle_results
gl_values = []
model_names = []
for gle_result, file_path in gle_results:
    gl_value = gle_result.metrics()["GL"]
    gl_values.append(gl_value)
    
    # Extract model name from file path
    model_name = file_path.split('/')[-2].split('_')[0]  # Extract model name from path
    model_names.append(model_name)

# Create a list of tuples and sort by model size (extracted from name)
def extract_size(model_name):
    """Extract numeric size from model name for sorting"""
    # Look for patterns like "7B", "8x7B", "70B", etc.
    size_match = re.search(r'(\d+(?:x\d+)?[BM])', model_name)
    if size_match:
        size_str = size_match.group(1)
        # Convert to comparable number (B = billion, M = million)
        if 'x' in size_str:
            # Handle cases like "8x7B"
            parts = size_str.replace('B', '').replace('M', '').split('x')
            return int(parts[0]) * int(parts[1])
        elif 'B' in size_str:
            return int(size_str.replace('B', ''))
        elif 'M' in size_str:
            return int(size_str.replace('M', '')) / 1000  # Convert to billions
    return 0  # Default for models without clear size indicator

# Combine data and sort by model size
model_data = list(zip(model_names, gl_values))
model_data.sort(key=lambda x: extract_size(x[0]))

# Separate back into lists
model_names_sorted = [item[0] for item in model_data]
gl_values_sorted = [item[1] for item in model_data]

# Create the plot
plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(model_names_sorted)), gl_values_sorted, color='skyblue', alpha=0.7)

# Customize the plot
plt.xlabel('Model (ordered by size)')
plt.ylabel('Grouping Loss (GL)')
plt.title('Grouping Loss Comparison Across Models (Ordered by Size)')
plt.xticks(range(len(model_names_sorted)), model_names_sorted, rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, gl_values_sorted)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# %%
