import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump
import os

# Define file paths
DATA_PATH = 'hospital_readmissions.csv'
MODEL_PATH = 'readmission_model.joblib'
METADATA_PATH = 'model_metadata.joblib'

def train_model():
    """
    Loads data, trains a Logistic Regression model for readmission prediction,
    and saves the model and its metadata to files.
    """
    print("--- Starting model training process ---")

    # 1. Load the dataset
    if not os.path.exists(DATA_PATH):
        print(f"Error: The data file '{DATA_PATH}' was not found.")
        return
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded successfully from '{DATA_PATH}'. Shape: {df.shape}")

    # 2. Select features and target
    features = [
        'time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications',
        'medical_specialty', 'change', 'diabetes_med'
    ]
    target = 'readmitted'

    # Drop rows with missing values in selected features for simplicity
    df.dropna(subset=features, inplace=True)
    
    # Map the target variable to binary (1 for 'yes', 0 for 'no')
    df['readmitted_binary'] = df[target].apply(lambda x: 1 if x == 'yes' else 0)

    X = df[features]
    y = df['readmitted_binary']

    # 3. Create a preprocessor for the features
    # Identify categorical and numerical features
    categorical_features = ['medical_specialty', 'change', 'diabetes_med']
    numerical_features = [
        'time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications'
    ]

    # Create the column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # 4. Create a machine learning pipeline
    # The pipeline will first preprocess the data and then apply the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear'))
    ])
    
    # 5. Train the model
    # Train on the entire dataset to ensure the model has seen all feature combinations
    print("Training the model...")
    model_pipeline.fit(X, y)
    print("Model training complete.")

    # Get the names of the final features after one-hot encoding
    final_feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()

    # 6. Save the trained model and metadata
    # Save the entire pipeline, including the preprocessor
    dump(model_pipeline, MODEL_PATH)

    # Save metadata needed for the Streamlit app
    metadata = {
        'preprocessor': model_pipeline.named_steps['preprocessor'],
        'medical_specialty_classes': sorted(df['medical_specialty'].unique()),
        'change_classes': sorted(df['change'].unique()),
        'diabetes_med_classes': sorted(df['diabetes_med'].unique()),
        'final_feature_names': final_feature_names
    }
    dump(metadata, METADATA_PATH)

    print(f"\nModel saved to '{MODEL_PATH}'")
    print(f"Metadata saved to '{METADATA_PATH}'")
    print("\nTraining process finished. You can now run your Streamlit app.")

if __name__ == '__main__':
    train_model()
