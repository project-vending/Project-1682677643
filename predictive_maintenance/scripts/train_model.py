python
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set path to load data
data_file = "../data/machine_data.csv"

# Load machine data into a Pandas DataFrame
machine_data = pd.read_csv(data_file)

# Clean and preprocess the data (not shown in this example)

# Split the data into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest classifier on the training data
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train_data, train_labels)

# Evaluate the model on the validation set
val_preds = rf.predict(val_data)
val_acc = np.mean(val_preds == val_labels)
print("Validation accuracy: {:.3f}".format(val_acc))

# Save the trained model to disk
model_file = "../models/predictive_model.pkl"
with open(model_file, "wb") as f:
    pickle.dump(rf, f)
