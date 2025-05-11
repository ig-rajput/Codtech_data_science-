import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Dummy dataset
data = {
    'age': [25, 30, None, 40, 35],
    'income': [50000, 60000, 55000, None, 65000],
    'gender': ['Male', 'Female', 'Female', 'Male', None],
    'purchased': [1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Split features and target
X = df.drop('purchased', axis=1)
y = df['purchased']

# Define preprocessing for numerical and categorical features
numerical_features = ['age', 'income']
categorical_features = ['gender']

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine pipelines
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Create full pipeline with model
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate model
model_pipeline.fit(X_train, y_train)
accuracy = model_pipeline.score(X_test, y_test)
print(f'Model accuracy: {accuracy:.2f}')
