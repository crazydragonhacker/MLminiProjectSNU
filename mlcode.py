# ------------------------------
# STEP 1: Import Libraries
# ------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ------------------------------
# STEP 2: Load Dataset from Google Drive
# ------------------------------
# Convert the shared link to a direct download link
url = "https://drive.google.com/uc?id=1RfDP3sOk6Ce5sIP-F1ayd3WsH0hwdEf9"
df = pd.read_csv(url)

print("✅ Data Loaded Successfully")
print(df.head())

# ------------------------------
# STEP 3: Clean & Prepare Data
# ------------------------------
# Rename columns for convenience (adjust if needed)
df = df.rename(columns={
    "ott_top1": "ott",
    "movie_genre_top1": "movie_genre",
    "series_genre_top1": "series_genre",
    "binge_freq_per_week": "binge_freq",
    "screen_time_movies_series_hours_per_we": "screen_time"
})

# Drop rows with missing values
df = df.dropna()

# ------------------------------
# STEP 4: Descriptive Analytics
# ------------------------------
# Find the King/Queen OTT
dominant_ott = df['ott'].value_counts().idxmax()
print(f"\n👑 OTT King/Queen of SNU = {dominant_ott}")

# Distribution of OTT platforms
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='ott', order=df['ott'].value_counts().index, palette="viridis")
plt.title("OTT Platform Dominance at SNU")
plt.xticks(rotation=30)
plt.show()

# Screen time per platform
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='ott', y='screen_time', palette="coolwarm")
plt.title("Screen Time Distribution by OTT Platform")
plt.xticks(rotation=30)
plt.show()

# ------------------------------
# STEP 5: Encode Categorical Features
# ------------------------------
le = LabelEncoder()
df['ott'] = le.fit_transform(df['ott'])
df['movie_genre'] = le.fit_transform(df['movie_genre'])
df['series_genre'] = le.fit_transform(df['series_genre'])

# Scale numerical features
scaler = StandardScaler()
df[['binge_freq','screen_time']] = scaler.fit_transform(df[['binge_freq','screen_time']])

# ------------------------------
# STEP 6: Train Predictive Model
# ------------------------------
X = df[['movie_genre','series_genre','binge_freq','screen_time']]
y = df['ott']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n📊 Model Evaluation Report:")
print(classification_report(y_test, y_pred))

# ------------------------------
# STEP 7: Feature Importance
# ------------------------------
importances = model.feature_importances_
feat_importance = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(6,4))
sns.barplot(data=feat_importance, x="Importance", y="Feature", palette="magma")
plt.title("Feature Importance in Predicting OTT Choice")
plt.show()