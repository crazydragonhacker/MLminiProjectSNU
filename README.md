# MLminiProjectSNU:
This is the project for SNU ML mini project.


# 🎬 OTT User Behavior Analysis

This project analyzes OTT (Over-The-Top) streaming user behavior using **Python** and **Machine Learning preprocessing techniques**.
We load survey data, preprocess it (encode labels, scale numeric features), and prepare it for further modeling (classification).

---

## 📂 Project Structure

```
.
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks with analysis
├── src/                   # Source code (preprocessing, utils, etc.)
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

---

## 🚀 Features

* Load dataset directly from Google Drive
* Clean & preprocess categorical and numerical features
* Label encode categorical columns like OTT platform, movie genre, series genre
* Scale numerical features like binge frequency per week
* Ready-to-use dataset for ML models

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ott-user-behavior.git
cd ott-user-behavior
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Usage

### 1. Load dataset from Google Drive

```python
import pandas as pd

file_id = "12xR8YWPMu7qQM-W-8GuykANQdzUEvajP"
downloadurl = [f"https://drive.google.com/uc?id={file_id}&export=download"](https://drive.google.com/uc?id=12xR8YWPMu7qQM-W-8GuykANQdzUEvajP&export=download)

df = pd.read_csv(url)
print(df.head())
```

### 2. Preprocess features

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

le = LabelEncoder()

# Encode categorical columns
df['Ott Top1'] = le.fit_transform(df['Ott Top1'])
df['Movie_genre_top1'] = le.fit_transform(df['Movie_genre_top1'])
df['Series_genre_top1'] = le.fit_transform(df['Series_genre_top1'])
df['Screen Time Movies/series in hours per week (Provide value between 0-40)'] = le.fit_transform(
    df['Screen Time Movies/series in hours per week (Provide value between 0-40)']
)

# Scale numeric features
scaler = StandardScaler()
df[['Binge frequency per week']] = scaler.fit_transform(df[['Binge frequency per week']])
```

---

## 🛠️ Tech Stack

* **Python 3.12+**
* **Pandas** – Data handling
* **Scikit-learn** – Preprocessing (LabelEncoder, StandardScaler)
* **gdown** – (optional) Google Drive file download helper

---

## 📌 To-Do

* [ ] Add ML models (classification)
* [ ] Perform EDA (Exploratory Data Analysis)
* [ ] Create visualizations for OTT user behavior patterns
* [ ] Deploy as a simple web app (Streamlit / Flask)

---

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.



