# 💼 Salary Prediction — Adult Census Income

A Machine Learning web application that predicts whether an individual's annual income exceeds **$50K** based on US Census data.

Built with **Python · pandas · scikit-learn · Streamlit**.

---

## 📁 Project Structure

```
SalaryPrediction/
├── app.py               # Streamlit web application
├── train_model.py       # Model training script
├── requirements.txt     # Python dependencies
├── README.md
├── agent_instructions.md
├── .env.example
├── data/
│   └── adult.csv        # UCI Adult Census Income dataset (48 842 rows)
└── models/
    ├── model.pkl        # Trained sklearn Pipeline (generated)
    └── feature_info.pkl # Feature metadata for the UI (generated)
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/SalaryPrediction.git
cd SalaryPrediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

This will:
- Load `data/adult.csv`
- Compare **Logistic Regression**, **Random Forest**, and **Gradient Boosting** using 5-fold cross-validation
- Select the best model by ROC-AUC
- Save `models/model.pkl` and `models/feature_info.pkl`

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Streamlit Community Cloud Deployment

1. Push this repository to GitHub (include the `data/` folder and `models/` folder with the `.pkl` files, **or** add a startup command that runs `train_model.py` before the app).
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repository, branch (`main`), and set the **Main file path** to `app.py`.
4. Click **Deploy**.

> **Tip:** To auto-train on cold start, add a `.streamlit/config.toml` or call `train_model.py` in a shell startup hook. The simplest approach is to commit the pre-generated `models/` files alongside the code.

---

## 🤖 ML Details

| Item | Value |
|---|---|
| Dataset | UCI Adult Census Income (48 842 samples) |
| Problem type | Binary Classification |
| Target column | `income` (`<=50K` / `>50K`) |
| Features | 14 (6 numeric + 8 categorical) |
| Models compared | Logistic Regression, Random Forest, Gradient Boosting |
| Selection metric | ROC-AUC (5-fold stratified CV) |
| Preprocessing | Median imputation (numeric), Mode imputation (categorical), StandardScaler, OrdinalEncoder |

---

## 📊 Features Used

| Feature | Type | Description |
|---|---|---|
| age | Numeric | Age of the individual |
| fnlwgt | Numeric | Census sampling weight |
| educational-num | Numeric | Education level (1–16) |
| capital-gain | Numeric | Capital gains ($) |
| capital-loss | Numeric | Capital losses ($) |
| hours-per-week | Numeric | Work hours per week |
| workclass | Categorical | Employment type |
| education | Categorical | Highest education attained |
| marital-status | Categorical | Marital status |
| occupation | Categorical | Job type |
| relationship | Categorical | Family relationship |
| race | Categorical | Race |
| gender | Categorical | Gender |
| native-country | Categorical | Country of origin |

---

## 📝 License

This project uses the [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/Adult) which is publicly available for research purposes.
