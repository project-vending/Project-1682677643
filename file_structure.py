 
import os

# Define root directory
root_dir = "predictive_maintenance"

# Define subdirectories
data_dir = os.path.join(root_dir, "data")
clean_data_dir = os.path.join(root_dir, "clean_data")
models_dir = os.path.join(root_dir, "models")
scripts_dir = os.path.join(root_dir, "scripts")

# Create subdirectories if they do not exist already
for dir in [data_dir, clean_data_dir, models_dir, scripts_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create empty files
open(os.path.join(data_dir, "machine_data.csv"), "w").close()
open(os.path.join(clean_data_dir, "clean_machine_data.csv"), "w").close()
open(os.path.join(models_dir, "predictive_model.pkl"), "w").close()
open(os.path.join(scripts_dir, "train_model.py"), "w").close()
open(os.path.join(scripts_dir, "predict.py"), "w").close()
