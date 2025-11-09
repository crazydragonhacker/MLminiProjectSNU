# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(page_title="BALA OTT ML ANALYSIS", layout="wide")
st.title("🎬 BALA OTT ML ANALYSIS")
st.markdown("### Machine Learning Mini Project – OTT Platform Prediction")

# --------------------------
# LINKS SECTION
# --------------------------
st.markdown("""
**📎 Project Resources**
- [Project Report (README)](https://github.com/crazydragonhacker/MLminiProjectSNU/blob/main/README.md)
- [Source Code (Notebook)](https://github.com/crazydragonhacker/MLminiProjectSNU/blob/main/Copy_of_MLminiProject.ipynb)
- ML App: *(This Streamlit App)*
""")

st.divider()

# --------------------------
# LOAD DATA
# --------------------------
DATA_URL = "https://drive.google.com/uc?id=12xR8YWPMu7qQM-W-8GuykANQdzUEvajP&export=download"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    return df

df_raw = load_data()
st.subheader("Raw Dataset (first 5 rows)")
st.dataframe(df_raw.head())

# --------------------------
# DATA CLEANING
# --------------------------
def preprocess_df(df):
    df = df.copy()
    df = df.rename(columns={
        "Ott Top1": "ott",
        "Movie_genre_top1": "movie_genre",
        "  Series_genre_top1  ": "series_genre",
        "  Binge frequency per week  ": "binge_freq",
        "  Screen Time Movies or series in hours per week.\n": "screen_time"
    })

    df = df.dropna(subset=['ott', 'movie_genre', 'series_genre'])
    df['binge_freq'] = pd.to_numeric(df['binge_freq'], errors='coerce')
    df['screen_time'] = pd.to_numeric(df['screen_time'], errors='coerce')
    df = df.dropna(subset=['binge_freq', 'screen_time'])

    for col in ['ott', 'movie_genre', 'series_genre']:
        df[col] = df[col].astype(str).str.strip()
    return df

df = preprocess_df(df_raw)
st.subheader("Cleaned Dataset (first 5 rows)")
st.dataframe(df.head())

# --------------------------
# ENCODING + SCALING
# --------------------------
le_ott = LabelEncoder()
le_movie = LabelEncoder()
le_series = LabelEncoder()

df['ott_enc'] = le_ott.fit_transform(df['ott'])
df['movie_enc'] = le_movie.fit_transform(df['movie_genre'])
df['series_enc'] = le_series.fit_transform(df['series_genre'])

scaler = StandardScaler()
df[['binge_scaled', 'screen_scaled']] = scaler.fit_transform(df[['binge_freq', 'screen_time']])

# --------------------------
# TRAIN MODEL
# --------------------------
X = df[['movie_enc', 'series_enc', 'binge_scaled', 'screen_scaled']]
y = df['ott_enc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("Model Evaluation")
st.text(classification_report(y_test, y_pred, target_names=le_ott.classes_))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6,4))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_ott.classes_).plot(ax=ax)
plt.title("Confusion Matrix")
st.pyplot(fig)

# --------------------------
# FEATURE IMPORTANCE
# --------------------------
feat_imp = pd.DataFrame({
    "Feature": ['Movie Genre', 'Series Genre', 'Binge Freq', 'Screen Time'],
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

st.subheader("Feature Importance")
fig2, ax2 = plt.subplots(figsize=(5,3))
sns.barplot(data=feat_imp, x="Importance", y="Feature", ax=ax2)
ax2.set_title("Feature Importance in OTT Prediction")
st.pyplot(fig2)

st.divider()

# --------------------------
# DESCRIPTIVE ANALYTICS
# --------------------------
st.header("📊 Descriptive Analytics")

col1, col2 = st.columns(2)
with col1:
    st.subheader("OTT Platform Popularity")
    fig3, ax3 = plt.subplots(figsize=(6,3))
    sns.countplot(data=df, x='ott', order=df['ott'].value_counts().index, ax=ax3)
    plt.xticks(rotation=30)
    st.pyplot(fig3)

with col2:
    st.subheader("Screen Time by OTT")
    fig4, ax4 = plt.subplots(figsize=(6,3))
    sns.boxplot(data=df, x='ott', y='screen_time', ax=ax4)
    plt.xticks(rotation=30)
    st.pyplot(fig4)

st.subheader("Summary Statistics")
st.dataframe(df.describe())

st.divider()

# --------------------------
# PREDICTION FORM
# --------------------------
st.header("🎯 Predict OTT Platform")

movie_options = sorted(df['movie_genre'].unique().tolist())
series_options = sorted(df['series_genre'].unique().tolist())

with st.form("predict_form"):
    colA, colB = st.columns(2)
    with colA:
        movie_in = st.selectbox("Movie Genre", movie_options)
        binge_in = st.number_input("Binge Frequency per Week", 0.0, 50.0, 3.0, step=1.0)
    with colB:
        series_in = st.selectbox("Series Genre", series_options)
        screen_in = st.number_input("Screen Time (hours/week)", 0.0, 100.0, 5.0, step=0.5)
    submitted = st.form_submit_button("Predict")

if submitted:
    movie_enc = le_movie.transform([movie_in])[0]
    series_enc = le_series.transform([series_in])[0]
    scaled_vals = scaler.transform([[binge_in, screen_in]])[0]
    user_input = np.array([[movie_enc, series_enc, scaled_vals[0], scaled_vals[1]]])
    pred_enc = model.predict(user_input)[0]
    pred_label = le_ott.inverse_transform([pred_enc])[0]
    st.success(f"Predicted Preferred OTT Platform: **{pred_label}**")

st.info("Developed by Subhroneel Bala | ML Mini Project")
