python
# Import required libraries
import pandas as pd
import pickle

# Load trained model from file
with open('../models/predictive_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load test data
test_data = pd.read_csv('../clean_data/clean_machine_data_test.csv')

# Make predictions
predictions = model.predict(test_data)

# Save predictions to file
predictions.to_csv('predictions.csv', index=False)
