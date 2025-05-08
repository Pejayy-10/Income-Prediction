import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import pickle

# 1. Data Loading and Initial Exploration
print("Loading dataset...")
# Define column names since the dataset doesn't include them
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 
    'marital_status', 'occupation', 'relationship', 'race', 'sex', 
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

# Load the dataset
df = pd.read_csv('adult.data', names=column_names, sep=', ', engine='python')
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData info:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe())

# 2. Data Cleaning
print("\n--- Data Cleaning ---")
# Check for missing values (in this dataset, missing values are marked as '?')
print("Missing values (marked as '?'):")
for column in df.columns:
    missing_count = (df[column] == ' ?').sum()
    if missing_count > 0:
        print(f"{column}: {missing_count}")

# Replace '?' with NaN and then handle missing values
for column in df.columns:
    df[column] = df[column].replace(' ?', np.nan)

# Handle missing values - for simplicity, we'll drop rows with missing values
# In a real project, you might want to use imputation instead
df_cleaned = df.dropna()
print(f"Shape after removing missing values: {df_cleaned.shape}")

# Check for duplicates
duplicates = df_cleaned.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
if duplicates > 0:
    df_cleaned = df_cleaned.drop_duplicates()
    print("Duplicates removed.")

# 3. Feature Engineering
print("\n--- Feature Engineering ---")
# Strip leading/trailing whitespace from string columns
for column in df_cleaned.select_dtypes(include=['object']).columns:
    df_cleaned[column] = df_cleaned[column].str.strip()

# Create age groups
df_cleaned['age_group'] = pd.cut(
    df_cleaned['age'], 
    bins=[0, 25, 35, 45, 55, 65, 100], 
    labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+']
)

# Create hours worked category
df_cleaned['work_intensity'] = pd.cut(
    df_cleaned['hours_per_week'], 
    bins=[0, 20, 40, 60, 100], 
    labels=['Part-time', 'Full-time', 'Overtime', 'Workaholic']
)

# Create a feature for education level (simplified)
education_map = {
    ' Preschool': 'Low',
    ' 1st-4th': 'Low',
    ' 5th-6th': 'Low',
    ' 7th-8th': 'Low',
    ' 9th': 'Low',
    ' 10th': 'Medium',
    ' 11th': 'Medium',
    ' 12th': 'Medium',
    ' HS-grad': 'Medium',
    ' Some-college': 'Medium',
    ' Assoc-voc': 'High',
    ' Assoc-acdm': 'High',
    ' Bachelors': 'High',
    ' Masters': 'Very High',
    ' Prof-school': 'Very High',
    ' Doctorate': 'Very High'
}
df_cleaned['education_level'] = df_cleaned['education'].map(education_map)

# 4. Target Preparation
print("\n--- Target Preparation ---")
# The target is already binary, but let's clean it up
df_cleaned['income'] = df_cleaned['income'].apply(lambda x: 1 if x == ' >50K' else 0)
print(f"Target variable distribution:\n{df_cleaned['income'].value_counts()}")

# 5. Feature Selection
print("\n--- Feature Selection ---")
# Identify categorical and numerical columns
categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove target from features
if 'income' in numerical_cols:
    numerical_cols.remove('income')

# Select K best features
X = df_cleaned.drop(columns=['income'])
y = df_cleaned['income']

# Using SelectKBest for numerical features
selector = SelectKBest(f_classif, k=min(10, len(numerical_cols)))
selector.fit(df_cleaned[numerical_cols], y)
selected_numerical = [numerical_cols[i] for i in selector.get_support(indices=True)]
print(f"Selected numerical features: {selected_numerical}")

# For categorical features, we'll select based on chi-squared test
# For simplicity, we'll keep all categorical features except those we created
selected_categorical = [col for col in categorical_cols if col not in ['education_level']]
print(f"Selected categorical features: {selected_categorical}")

selected_features = selected_numerical + selected_categorical
print(f"Total selected features: {len(selected_features)}")

# 6. Data Processing
print("\n--- Data Processing ---")
# Split the data
X = df_cleaned[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

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
# Define models to compare
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
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