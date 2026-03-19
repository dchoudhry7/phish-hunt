# 🔐 PhishHunt – AI-Based Phishing Detection System

## 📌 Overview

PhishHunt is a machine learning-based web application that detects whether a given URL is **legitimate or phishing**. It analyzes multiple URL-based features and uses a trained ML model to classify threats in real time.

---

## 🚀 Features

* 🔍 URL-based phishing detection
* 🤖 Machine Learning model (Scikit-learn)
* 🌐 Flask web interface for real-time predictions
* 📊 Feature extraction from URLs
* ⚡ Fast and lightweight

---

## 🧠 Tech Stack

* **Frontend:** HTML, CSS, Bootstrap
* **Backend:** Flask (Python)
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Other Tools:** BeautifulSoup, Requests, tldextract

---

## 📂 Project Structure

```
PhishHunt/
│── app.py
│── model.pkl
│── templates/
│── static/
│── utils/
│── requirements.txt
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/YOUR_USERNAME/phishhunt-ai.git
cd phishhunt-ai
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run the application

```
python app.py
```

### 5. Open in browser

```
http://127.0.0.1:5000/
```

---

## 📊 How It Works

1. User enters a URL
2. System extracts features (domain age, HTTPS usage, length, etc.)
3. Features are passed to trained ML model
4. Model predicts → **Phishing or Legitimate**

---

## 📸 Screenshots

(Add screenshots here – VERY important for resume)

---

## 🎯 Future Improvements

* Real-time browser extension
* Integration with threat intelligence APIs
* Deep learning-based detection
* Deploy on cloud (Render / AWS)

---

## 👨‍💻 Author

Dhairya Choudhry

* GitHub: https://github.com/dchoudhry7
* LinkedIn: www.linkedin.com/in/dchoudhry7

---
