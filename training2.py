import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Load data
data = pd.read_csv('Student Mental health.csv')

# Data Cleaning (Example of basic cleaning)
data.dropna(inplace=True)  # Remove any rows with missing values

# Splitting data into features and target
X = data.drop(['Do you have Depression?'], axis=1)
y = data['Do you have Depression?'].apply(lambda x: 1 if x == 'Yes' else 0)

# Preprocessing pipeline
numeric_features = ['Age']  # Replace with all numerical column names if more
categorical_features = ['Choose your gender', 'What is your course?', 'Your current year of Study', 'Marital status']

# Create transformers for numerical and categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')  # Ignore unknown categories in test data

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define SVM model
svm_model = SVC()

# Create a pipeline that includes preprocessing and model training
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', svm_model)])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy}")

# Save the model with pickle
with open('best_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print("SVM model saved as best_model.pkl")
