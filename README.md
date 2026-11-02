# 🏦 Bank Fraud Detection System Using Machine Learning

An intelligent machine learning application designed to identify potentially fraudulent bank transactions and distinguish them from genuine transactions.

## 📌 Project Overview

Bank fraud is a major challenge in digital banking. This project uses Machine Learning to analyze transaction data and detect suspicious activities.

The system is built using Python and Streamlit, providing an interactive dashboard for transaction analysis, fraud prediction, and model performance evaluation.

## 🎯 Objectives

* Detect fraudulent financial transactions.
* Classify transactions as Genuine or Fraudulent.
* Analyze transaction patterns using data visualization.
* Evaluate machine learning model performance.
* Provide an interactive and user-friendly dashboard.

## ✨ Key Features

* Interactive dashboard with transaction statistics.
* Fraud and genuine transaction analysis.
* Machine Learning-based fraud prediction.
* Fraud probability visualization.
* Model performance evaluation.
* Confusion matrix and classification report.
* Interactive charts and graphs.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Plotly
* Matplotlib
* Seaborn
* Imbalanced-learn
* Joblib

## 🧠 Machine Learning Model

**Algorithm:** Random Forest Classifier

The model is trained using the Credit Card Fraud Detection dataset. It learns transaction patterns and predicts whether a transaction is potentially fraudulent or genuine.

## 📂 Project Structure

```text
Bank_Fraud_Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
│   └── style.css
│
├── data/
│   └── creditcard.csv
│
├── model/
│   ├── fraud_model.pkl
│   └── scaler.pkl
│
├── pages/
│   ├── dashboard.py
│   ├── prediction.py
│   └── reports.py
│
└── src/
    ├── preprocessing.py
    ├── prediction.py
    ├── evaluation.py
    └── train_model.py
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/laxmiramappapalled/Bank-Fraud-Detection.git
```

### 2. Navigate to the Project Folder

```bash
cd Bank-Fraud-Detection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the Credit Card Fraud Detection dataset from Kaggle.

Place the downloaded `creditcard.csv` file inside the `data` folder.

### 5. Train the Model

```bash
python src/train_model.py
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Dataset Information

Dataset: Credit Card Fraud Detection

* Total Transactions: 284,807
* Genuine Transactions: 284,315
* Fraudulent Transactions: 492
* Total Features: 30 input features and 1 target column

**Note:** The dataset is not included in this GitHub repository because of its large file size.

## ⚠️ Disclaimer

This project is developed for educational and demonstration purposes. Its predictions should not be used as the sole basis for real-world financial decisions.

## 👩‍💻 Developed By

**Laxmi Ramappa Palled**

Diploma in Computer Science and Engineering

GitHub: [laxmiramappapalled](https://github.com/laxmiramappapalled)

---

⭐ If you find this project interesting, feel free to explore the repository!
