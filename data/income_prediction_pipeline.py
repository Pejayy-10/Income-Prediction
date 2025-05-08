import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# 1. Data Loading and Initial Exploration
print("Loading dataset...")
# Define column names since the dataset doesn't include them
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 
    'marital_status', 'occupation', 'relationship', 'race', 'sex', 
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

# Load the dataset
try:
    # Try to load with comma-space separator
    df = pd.read_csv('adult.data', names=column_names, sep=', ', engine='python')
except:
    # If that fails, try with comma separator
    df = pd.read_csv('adult.data', names=column_names, sep=',', engine='python')

print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# 2. Data Cleaning
print("\n--- Data Cleaning ---")
# Check for missing values (in this dataset, missing values are marked as '?')
print("Checking for missing values (marked as '?'):")
missing_counts = {}
for column in df.columns:
    # Check for both ' ?' and '?' values
    missing_count = (df[column] == ' ?').sum() + (df[column] == '?').sum()
    if missing_count > 0:
        missing_counts[column] = missing_count
        print(f"{column}: {missing_count}")

# Replace '?' with NaN and then handle missing values
for column in df.columns:
    df[column] = df[column].replace(' ?', np.nan).replace('?', np.nan)

# For categorical columns with missing values, replace with the most frequent value
for column in df.select_dtypes(include=['object']).columns:
    if df[column].isna().sum() > 0:
        most_frequent = df[column].mode()[0]
        df[column].fillna(most_frequent, inplace=True)

# For numerical columns with missing values, replace with the median
for column in df.select_dtypes(include=['int64', 'float64']).columns:
    if df[column].isna().sum() > 0:
        median_value = df[column].median()
        df[column].fillna(median_value, inplace=True)

print(f"Shape after handling missing values: {df.shape}")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")

# 3. Feature Engineering
print("\n--- Feature Engineering ---")
# Strip leading/trailing whitespace from string columns
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].str.strip()

# Create age groups
df['age_group'] = pd.cut(
    df['age'], 
    bins=[0, 25, 35, 45, 55, 65, 100], 
    labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+']
)

# Create hours worked category
df['work_intensity'] = pd.cut(
    df['hours_per_week'], 
    bins=[0, 20, 40, 60, 100], 
    labels=['Part-time', 'Full-time', 'Overtime', 'Workaholic']
)

# Create a feature for education level (simplified)
education_map = {
    'Preschool': 'Low',
    '1st-4th': 'Low',
    '5th-6th': 'Low',
    '7th-8th': 'Low',
    '9th': 'Low',
    '10th': 'Medium',
    '11th': 'Medium',
    '12th': 'Medium',
    'HS-grad': 'Medium',
    'Some-college': 'Medium',
    'Assoc-voc': 'High',
    'Assoc-acdm': 'High',
    'Bachelors': 'High',
    'Masters': 'Very High',
    'Prof-school': 'Very High',
    'Doctorate': 'Very High'
}

# Apply education mapping, handling potential missing keys
df['education_level'] = df['education'].apply(
    lambda x: education_map.get(x, 'Medium') if pd.notna(x) else 'Medium'
)

# 4. Target Preparation
print("\n--- Target Preparation ---")
# The target is already binary, but let's clean it up
# Check unique values in income column
print(f"Unique values in income column: {df['income'].unique()}")

# Clean up the income column - handle different possible formats
df['income'] = df['income'].apply(
    lambda x: 1 if ('>50K' in str(x) or '>50k' in str(x).lower()) else 0
)

print(f"Target variable distribution:\n{df['income'].value_counts()}")
print(f"Target variable distribution (percentage):\n{df['income'].value_counts(normalize=True) * 100}")

# Check if we have both classes
if len(df['income'].unique()) < 2:
    raise ValueError("Target variable has less than 2 classes. Check your data preprocessing steps.")

# 5. Feature Selection
print("\n--- Feature Selection ---")
# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove target from features
if 'income' in numerical_cols:
    numerical_cols.remove('income')

# Select K best features
X = df.drop(columns=['income'])
y = df['income']

# Using SelectKBest for numerical features
if len(numerical_cols) > 0:
    selector = SelectKBest(f_classif, k=min(10, len(numerical_cols)))
    selector.fit(df[numerical_cols], y)
    selected_numerical = [numerical_cols[i] for i in selector.get_support(indices=True)]
    print(f"Selected numerical features: {selected_numerical}")
else:
    selected_numerical = []

# For categorical features, we'll select based on chi-squared test
# For simplicity, we'll keep all categorical features except those we created
selected_categorical = [col for col in categorical_cols if col not in ['education_level']]
print(f"Selected categorical features: {selected_categorical}")

selected_features = selected_numerical + selected_categorical
print(f"Total selected features: {len(selected_features)}")

# 6. Data Processing
print("\n--- Data Processing ---")
# Split the data - use stratify to maintain class distribution
X = df[selected_features]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Training set target distribution: {pd.Series(y_train).value_counts(normalize=True) * 100}")
print(f"Testing set target distribution: {pd.Series(y_test).value_counts(normalize=True) * 100}")

# Create preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, selected_numerical),
        ('cat', categorical_transformer, selected_categorical)
    ])

# 7. Model Training and Comparison
print("\n--- Model Training and Comparison ---")
# Define models to compare - removing Gradient Boosting which was causing issues
models = {
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
}

best_accuracy = 0
best_model = None
best_model_name = None

for name, model in models.items():
    # Create pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    print(f"Training {name}...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline
        best_model_name = name

print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")

# 8. Save Model
print("\n--- Saving Model ---")
if best_accuracy >= 0.8:  # Check if accuracy meets requirement
    # Save the model using pickle
    with open('income_prediction_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    print("Model saved successfully with accuracy above 80%!")
    
    # Also save the feature list for future use
    with open('selected_features.pkl', 'wb') as file:
        pickle.dump(selected_features, file)
    print("Selected features saved.")
    
    # Save the preprocessor for future use
    with open('preprocessor.pkl', 'wb') as file:
        pickle.dump(preprocessor, file)
    print("Preprocessor saved.")
else:
    print(f"Model accuracy ({best_accuracy:.4f}) is below 80%. Consider improving the model.")

# 9. Test Predictions
print("\n--- Test Predictions ---")
# Make predictions on a few test samples
sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
samples = X_test.iloc[sample_indices]
true_labels = y_test.iloc[sample_indices]
predictions = best_model.predict(samples)

print("Sample predictions:")
for i, (idx, sample) in enumerate(samples.iterrows()):
    print(f"Sample {i+1}:")
    for feature, value in sample.items():
        print(f"  {feature}: {value}")
    print(f"  True label: {'Income >50K' if true_labels.iloc[i] == 1 else 'Income <=50K'}")
    print(f"  Predicted label: {'Income >50K' if predictions[i] == 1 else 'Income <=50K'}")
    print()

print("Data science pipeline completed!")