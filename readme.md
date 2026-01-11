# 🩺 Healix AI | Intelligent Healthcare Diagnosis System

> **AI-powered disease prediction and personalized health assistant.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini-4285F4)

## 📖 Overview

**Healix AI** is a machine learning-based healthcare application designed to assist users in identifying potential health conditions based on their symptoms. Beyond simple diagnosis, it provides a holistic health report including:
* **Predicted Disease** with confidence scores.
* **Recommended Medications** (Informational only).
* **Dietary Advice** tailored to the condition.
* **Workout Plans** to aid recovery.
* **Precautionary Measures** to prevent spread or worsening.

The application also features an **Integrated AI Chatbot** (powered by Google Gemini) to answer follow-up health questions in a conversational manner.

---

## ✨ Key Features

* **🤖 ML Symptom Analysis:** Uses a trained **Support Vector Classifier (SVC)** model to predict diseases from a dataset of 130+ symptoms.
* **💬 Interactive AI Assistant:** Seamlessly chat with Healix AI (powered by **Google Gemini Flash**) for deeper insights into your diagnosis.
* **🛡️ Safety First:** Includes logic to detect critical symptoms (e.g., severe urinary issues) and advise immediate medical attention instead of a generic prediction.
* **⚡ Real-Time UX:** Features a responsive design, "Thinking..." states for AI interactions, and voice input support.
* **📊 Comprehensive Reports:** Generates structured "Bento Grid" style reports for easy reading.

---

## 🛠️ Tech Stack

* **Backend:** Python, Flask, Gunicorn
* **Frontend:** HTML5, CSS3, JavaScript, Bootstrap 5
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **AI Integration:** Google Generative AI (Gemini API)
* **Deployment:** Ready for Render / Railway

---

## 🚀 Installation & Local Setup

Follow these steps to run Healix AI on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/ZahedShaikh08/HealixAI.git](https://github.com/ZahedShaikh08/HealixAI.git)
cd HealixAI
```
### 2. Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Configure Environment Variables
Create a .env file in the root directory and add your API keys:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
FLASK_SECRET_KEY=your_random_secret_key
```
### 5. Run the Application
```bash
python main.py
```
Open your browser and visit: http://127.0.0.1:5000

---

## 📂 Project Structure

```bash
HealixAI/
├── Datasets/          # CSV files (symptoms, precautions, diets, etc.)
├── models/            # Trained Machine Learning models (svc.pkl)
├── static/            # Frontend assets (CSS, Images, JS)
├── templates/         # HTML templates for Flask
├── main.py            # Main application entry point
├── requirements.txt   # Python dependencies list
├── Procfile           # Deployment configuration for Render
├── render.yaml        # Infrastructure as Code for Render
└── README.md          # Project documentation
```
---

## ⚠️ Disclaimer
Healix AI is for informational purposes only. It does not provide medical advice, diagnosis, or treatment. The predictions are based on a machine learning model and should not replace professional medical consultation. Always see a doctor for serious health concerns.

---

## 👨‍💻 Author
Zahed Shaikh

* **Role:** AI/ML Engineer
* **GitHub:** ZahedShaikh08