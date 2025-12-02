import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv('matches.csv')

# Data cleaning
df_clean = df.drop(['umpire1', 'umpire2', 'umpire3', 'player_of_match'], axis=1)
df_clean = df_clean.dropna(subset=['winner'])
df_clean['city'] = df_clean['city'].fillna('Unknown')

# Label Encoding
categorical_cols = ['team1', 'team2', 'city', 'venue', 'toss_winner', 'toss_decision', 'winner']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

# Feature selection
features = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']
target = 'winner'
X = df_clean[features]
y = df_clean[target]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model building
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and label encoders
joblib.dump(model, 'cricket_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
print("Model and encoders saved.")
# Optional: Print accuracy
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Save the clean and encoded DataFrame to a new CSV file
df_clean.to_csv('cleaned_and_encoded_matches.csv', index=False)
print("Cleaned and encoded dataset saved to 'cleaned_and_encoded_matches.csv'")
