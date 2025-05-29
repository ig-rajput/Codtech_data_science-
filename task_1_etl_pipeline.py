import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('sample_data/data.csv')

# Encode target if categorical
if df[df.columns[-1]].dtype == 'object':
    le = LabelEncoder()
    df[df.columns[-1]] = le.fit_transform(df[df.columns[-1]])

# Split features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save transformed data
pd.DataFrame(X_train).to_csv('sample_data/X_train.csv', index=False)
pd.DataFrame(X_test).to_csv('sample_data/X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('sample_data/y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('sample_data/y_test.csv', index=False)

print("ETL pipeline completed and data saved.")
