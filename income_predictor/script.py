# setup_directories.py
import os

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define directories to create
directories = [
    os.path.join(BASE_DIR, 'static'),
    os.path.join(BASE_DIR, 'predictor', 'static'),
    os.path.join(BASE_DIR, 'predictor', 'static', 'predictor'),
    os.path.join(BASE_DIR, 'predictor', 'static', 'predictor', 'css'),
    os.path.join(BASE_DIR, 'predictor', 'static', 'predictor', 'js'),
    os.path.join(BASE_DIR, 'predictor', 'templates'),
    os.path.join(BASE_DIR, 'predictor', 'templates', 'predictor'),
    os.path.join(BASE_DIR, 'predictor', 'ml_models'),
]

# Create directories if they don't exist
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

print("Directory setup complete!")