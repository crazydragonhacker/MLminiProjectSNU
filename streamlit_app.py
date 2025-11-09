# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import qrcode
from io import BytesIO
import numpy as np

st.set_page_config(page_title="BALA OTT ML ANALYSIS", layout="wide")

# -------------------------
# User-supplied links & title
# -------------------------
PROJECT_REPORT_URL = "https://github.com/crazydragonhacker/MLminiProjectSNU/blob/main/README.md"
CODE_URL = "https://github.com/crazydragonhacker/MLminiProjectSNU/blob/main/Copy_of_MLminiProject.ipynb"
# placeholder — replace with deployed app URL once you publish
ML_APP_URL = "https://(paste-your-deployed-streamlit-link-here)"

APP_TITLE = "BALA OTT ML ANALYSIS"

# -------------------------
# Helper: QR code generator
# -------------------------
def make_qr_bytes(url: str, box_size: int = 6) -> bytes:
    qr = qrcode.QRCode(box_size=box_size, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# -------------------------
# Header / Resource links
# -------------------------
st.title(f"🎬 {APP_TITLE}")
st.markdown("A Streamlit interface for the ML mini-project predicting users' preferred OTT platform.")

st.header("📎 Project Resources")
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.image(make_qr_bytes(PROJECT_REPORT_URL), width=140)
    st.markdown(f"**Project Report**  \n[Open README]({PROJECT_REPORT_URL})")
with col2:
    st.image(make_qr_bytes(CODE_URL), width=140)
    st.markdown(f"**Source Code**  \n[Open Notebook]({CODE_URL})")
with col3:
    st.image(make_qr_bytes(ML_APP_URL), width=140)
    st.markdown(f"**ML App**  \n[Paste deployed app URL above]({ML_APP_URL})")

st.markdown("---")

# -------------------------
# Load data (same link used in provided file)
# -------------------------
DATA_URL = "https://drive.google.com/uc?id=12xR8YWPMu7qQM-W-8GuykANQdzUEvajP&export=download"

@st.cache_data
def load_data(url=DATA_URL):
    df = pd.read_csv(url)
    return df

df_raw = load_data()
st.subheader("Raw Data (first 5 rows)")
st.dataframe(df_raw.head())

# -------------------------
# Preprocessing (adapted from original file)
# -------------------------
def preprocess_df(df):
    df = df.copy()

    df = df.rename(columns={
        "Ott Top1": "ott",
        "Movie_genre_top1": "movie_genre",
        "  Series_genre_top1  ": "series_genre",
        "  Binge frequency per week  ": "binge_freq",
        "  Screen Time Movies or series in hours per week.\n": "screen_time"
    })

    # drop rows with missing ott or genre columns
    df = df.dropna(subset=['ott','movie_genre','series_genre'])

    # coerce numeric
    df['binge_freq'] = pd.to_numeric(df['binge_freq'], errors='coerce')
    df['screen_time'] = pd.to_numeric(df['screen_time'], errors='coerce')

    df = df.dropna(subset=['binge_freq','screen_time'])

    # strip strings (in case)
    df['movie_genre'] = df['movie_genre'].astype(str).str.strip()
    df['series_genre'] = df['series_genre'].astype(str).str.strip()
    df['ott'] = df['ott'].astype(str).str.strip()

    return df

df = preprocess_df(df_raw)

st.subheader("Cleaned Data (first 5 rows)")
st.dataframe(df.head())

# -------------------------
# Encoders and scaling
# -------------------------
# Use separate LabelEncoders for each categorical column
le_ott = LabelEncoder()
le_movie = LabelEncoder()
le_series = LabelEncoder()

df['ott_enc'] = le_ott.fit_transform(df['ott'])
df['movie_genre_enc'] = le_movie.fit_transform(df['movie_genre'])
df['series_genre_enc'] = le_series.fit_transform(df['series_genre'])

scaler = StandardScaler()
df[['binge_freq_s','screen_time_s']] = scaler.fit_transform(df[['binge_freq','screen_time']])

# Prepare X and y
X = df[['movie_genre_enc','series_genre_enc','binge_freq_s','screen_time_s']]
y = df['ott_enc']

# Train/test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("Model Evaluation")
st.text("Classification report for the RandomForestClassifier (test set):")
report = classification_report(y_test, y_pred, output_dict=False)
st.text(report)

# show confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(6,4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_ott.classes_)
disp.plot(ax=ax_cm, xticks_rotation=45)
plt.title("Confusion Matrix")
st.pyplot(fig_cm)

# Feature importance
importances = model.feature_importances_
feat_importance = pd.DataFrame({
    "Feature": ['movie_genre','series_genre','binge_freq','screen_time'],
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.subheader("Feature Importance")
fig_fi, ax_fi = plt.subplots(figsize=(6,3))
sns.barplot(data=feat_importance, x="Importance", y="Feature", ax=ax_fi)
ax_fi.set_title("Feature Importance")
st.pyplot(fig_fi)

st.markdown("---")

# -------------------------
# Descriptive plots
# -------------------------
st.header("Descriptive Analytics")

cola, colb = st.columns(2)
with cola:
    st.subheader("OTT Platform Counts")
    fig1, ax1 = plt.subplots(figsize=(6,3))
    order = df['ott'].value_counts().index
    sns.countplot(data=df, x='ott', order=order, ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30)
    st.pyplot(fig1)

with colb:
    st.subheader("Screen Time by OTT (boxplot)")
    fig2, ax2 = plt.subplots(figsize=(6,3))
    sns.boxplot(data=df, x='ott', y='screen_time', ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30)
    st.pyplot(fig2)

st.subheader("Data Summary")
st.write(df[['ott','movie_genre','series_genre','binge_freq','screen_time']].describe(include='all'))

st.markdown("---")

# -------------------------
# Interactive prediction form
# -------------------------
st.header("🧪 Predict Preferred OTT Platform")

st.markdown("Enter a user profile and click **Predict** to see the model's prediction.")

# options for dropdowns built from the cleaned df (original string values)
movie_options = sorted(df['movie_genre'].unique().tolist())
series_options = sorted(df['series_genre'].unique().tolist())

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        movie_in = st.selectbox("Movie Genre (top1)", movie_options)
        binge_in = st.number_input("Binge frequency per week", min_value=0.0, max_value=100.0, value=3.0, step=1.0)
    with col2:
        series_in = st.selectbox("Series Genre (top1)", series_options)
        screen_in = st.number_input("Screen time (hours per week)", min_value=0.0, max_value=168.0, value=5.0, step=0.5)
    submitted = st.form_submit_button("Predict")

if submitted:
    # encode inputs using trained LabelEncoders & scale numeric values
    try:
        movie_enc = le_movie.transform([movie_in])[0]
        series_enc = le_series.transform([series_in])[0]
    except Exception as e:
        st.error("Selected genre not recognized by the encoder. This can happen if the app was restarted and encoders were not refit. Please reload the app.")
        raise e

    num_scaled = scaler.transform([[binge_in, screen_in]])[0]  # returns [binge_freq_s, screen_time_s]
    X_user = np.array([[movie_enc, series_enc, num_scaled[0], num_scaled[1]]])
    pred_enc = model.predict(X_user)[0]
    pred_label = le_ott.inverse_transform([pred_enc])[0]

    st.success(f"Predicted preferred OTT platform: **{pred_label}**")

    st.markdown("**Model details**")
    st.write({
        "movie_genre_enc": int(movie_enc),
        "series_genre_enc": int(series_enc),
        "binge_freq (raw)": binge_in,
        "screen_time (raw)": screen_in
    })

st.markdown("---")

# -------------------------
# Extras: show mapping tables
# -------------------------
st.header("Mappings & Notes")
st.subheader("Label encodings (sample)")
enc_map_movie = pd.DataFrame({
    "movie_genre": le_movie.classes_,
    "movie_genre_enc": range(len(le_movie.classes_))
})
enc_map_series = pd.DataFrame({
    "series_genre": le_series.classes_,
    "series_genre_enc": range(len(le_series.classes_))
})
enc_map_ott = pd.DataFrame({
    "ott": le_ott.classes_,
    "ott_enc": range(len(le_ott.classes_))
})

c1, c2, c3 = st.columns(3)
with c1:
    st.write("Movie genre encoding")
    st.dataframe(enc_map_movie)
with c2:
    st.write("Series genre encoding")
    st.dataframe(enc_map_series)
with c3:
    st.write("OTT encoding")
    st.dataframe(enc_map_ott)

st.markdown("---")
st.info("To publish your app: 1) push this file to GitHub, 2) Deploy on Streamlit Community Cloud (share.streamlit.io) and paste the app URL above so the QR code updates.")
