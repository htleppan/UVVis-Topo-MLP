import sys
sys.path.append('/usr/local/lib/python3.8/site-packages/')

import numpy as np
import pandas as pd 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

### Read data
input_data = np.genfromtxt ('input_data.csv', delimiter=",")
output_data= np.genfromtxt ('output_data.csv', delimiter=",")
smiles_data = pd.read_csv("smiles.csv").iloc[:, 1]

### Split to train and test sets
X_train, X_test, y_train, y_test, smi_train, smi_test = train_test_split(input_data, output_data, smiles_data, test_size=0.4, random_state=42)
print("X_train " + str(X_train.shape))
print("X_test " + str(X_test.shape))


### Discard features with entries in only one row
retained_features = np.sum(X_train > 0, axis = 0) > 1 
X_train = X_train[:, retained_features]
X_test = X_test[:, retained_features]



### Naive oversampling: divide y range to n_bins and fill all bins to max by resampling from each bin

oversampling = False
if oversampling:
    n_bins = 20
    y_bin_labels = np.round((y_train - np.min(y_train)) / (np.max(y_train) - np.min(y_train)) * n_bins)
    y_bins_count = [np.sum(y_bin_labels == y) for y in range(0, n_bins)]
    for y_bin in range(0, n_bins):
       if y_bins_count[y_bin] == 0:	# Cannot sample from an empty bin
          continue
       # Number of entries to be sampled
       n_to_resample = max(y_bins_count) - y_bins_count[y_bin]

       # Bin population
       y_bin_idxs = np.where(y_bin_labels == y_bin)[0]
       # Sample from the bin population
       sampled_idxs = np.random.choice(y_bin_idxs, n_to_resample)

       # Append to training data
       y_train = np.concatenate((y_train, y_train[sampled_idxs]))
       X_train = np.vstack((X_train, X_train[sampled_idxs,:]))


### MLP Regression
reg = MLPRegressor(random_state=11, hidden_layer_sizes=(round(X_train.shape[1]*0.6)), max_iter=5000, verbose = True)
reg.fit(X_train, y_train)

print("Train score: " + str(reg.score(X_train, y_train)))
print("Test score:  " + str(reg.score(X_test, y_test)))

yhat_test = reg.predict(X_test)

### Calculate outliers (+- 50 nm from target)
outliers = []
for y in range(0, yhat_test.shape[0]):
    if np.abs(yhat_test[y] - y_test[y]) > 50:
        outliers.append([y_test[y], yhat_test[y], smi_test.iloc[y]])
print("Outliers: " + str(len(outliers)))






