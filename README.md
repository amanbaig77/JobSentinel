# 🛡️ JobSentinel

### Machine Learning Powered Job Fraud Detection & Real-Time Job Recommendation System

JobSentinel is a machine-learning based application designed to help job seekers identify potentially fraudulent job postings and discover similar real-world job opportunities.

The system combines **Natural Language Processing (NLP)**, **machine learning classification**, **risk analysis**, **text similarity**, and the **Adzuna Jobs API** inside an interactive **Streamlit** application.

The project was built to address a practical problem faced by job seekers: **How can we identify suspicious job postings before applying, while also finding legitimate alternatives that match the same career opportunity?**

---

## 🚀 Live Demo

🔗 **Streamlit App:**
`https://jobsentinelgit-fxeawaw2cypbbh5wmcxjbs.streamlit.app`

🔗 **GitHub Repository:**
https://github.com/amanbaig77/JobSentinel

---

# 📌 Project Overview

Online job platforms contain thousands of job advertisements, but not every posting is trustworthy.

Fraudulent job postings can contain suspicious descriptions, unrealistic requirements, missing information, unusual contact details, or other characteristics associated with scam listings.

JobSentinel provides a two-part solution:

### 1. 🔍 Job Fraud Detection

A machine-learning model analyzes the submitted job posting and predicts whether it is:

* **LEGITIMATE**
* **FRAUDULENT**

The system also produces:

* Model decision score
* Decision threshold
* Risk level
* Explainable risk signals
* Interpretation of the prediction

### 2. 💼 Similar Job Recommendations

After analyzing a job posting, JobSentinel can search the **Adzuna Jobs API** and retrieve real job opportunities related to the submitted:

* Job title
* Location

The retrieved jobs are normalized and presented as similar opportunities.

---

# ✨ Key Features

## 🛡️ Machine Learning Fraud Detection

* Binary job fraud classification
* Linear SVM / LinearSVC-based model
* TF-IDF text representation
* Categorical feature encoding
* Numerical feature scaling
* Custom decision threshold
* Risk-level interpretation
* Explainable risk signals

## 🧠 NLP-Based Feature Engineering

The system extracts multiple characteristics from job postings, including:

* Text length
* Word count
* URL count
* Email count
* Phone-number count
* Uppercase ratio
* Suspicious-term frequency
* Combined job-posting text

## 📊 Risk Assessment

JobSentinel converts the machine-learning prediction into an easier-to-understand risk assessment.

The application communicates:

* Prediction
* Decision score
* Threshold
* Risk level
* Supporting signals

This makes the model output more understandable than simply displaying `0` or `1`.

## 🔎 Real-Time Job Search

JobSentinel integrates with the **Adzuna Jobs API** to retrieve real job listings.

Users can search based on:

* Job title
* Location

For example:

```text
Job Title: Python Developer
Location: Bangalore
```

The application can retrieve relevant jobs such as:

* Backend Developer – Python
* Senior Python Full Stack Developer
* Python Developer
* Other related opportunities

## 🔗 Job Recommendation

Retrieved job listings are normalized into a consistent internal format containing information such as:

* Job title
* Company
* Location
* Salary
* Contract type
* Contract time
* Category
* Description
* Created date
* Job URL

## 🌐 Streamlit Web Application

The entire system is available through a browser-based Streamlit interface.

Users do not need to interact with Python scripts or machine-learning models directly.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │     User Input      │
                    │                     │
                    │ Job Title           │
                    │ Company Profile     │
                    │ Description         │
                    │ Requirements        │
                    │ Benefits            │
                    │ Location            │
                    │ Salary              │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Preparation  │
                    │                     │
                    │ Cleaning            │
                    │ Missing values      │
                    │ Combined text       │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │      Feature Engineering       │
              │                                │
              │ TF-IDF                         │
              │ Categorical Encoding           │
              │ Numerical Features             │
              │ Suspicious Terms               │
              │ URL / Email / Phone Counts     │
              └────────────────┬───────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   ML Model          │
                    │                     │
                    │ Linear SVM          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Fraud Prediction    │
                    │                     │
                    │ Legitimate / Fraud  │
                    │ Risk Level          │
                    │ Decision Score      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Adzuna Job Search   │
                    │                     │
                    │ Title + Location    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Similar Jobs        │
                    │                     │
                    │ Company             │
                    │ Location            │
                    │ Salary              │
                    │ Description         │
                    │ Apply Link          │
                    └─────────────────────┘
```

---

# 🧠 Machine Learning Pipeline

The prediction pipeline is designed so that the same preprocessing objects used during model development are reused during inference.

The saved model components include:

```text
models/
├── best_model.joblib
├── tfidf_vectorizer.joblib
├── categorical_encoder.joblib
├── numeric_scaler.joblib
├── feature_config.json
└── decision_threshold.txt
```

This prevents inconsistencies between training-time and prediction-time preprocessing.

---

# 🔤 Text Processing

JobSentinel combines important textual fields from the job posting into a unified text representation.

Typical text fields include:

* Job title
* Company profile
* Description
* Requirements
* Benefits

The combined text is cleaned and normalized before being passed into the TF-IDF vectorizer.

---

# 📐 Engineered Numerical Features

In addition to TF-IDF features, JobSentinel extracts behavioral and structural characteristics from the text.

### Text length

Measures the number of characters contained in a text field.

### Word count

Measures the number of words in the field.

### URL count

Detects URLs contained in the job posting.

### Email count

Detects email addresses.

### Phone-number count

Detects potential phone numbers.

### Uppercase ratio

Measures the proportion of alphabetic characters written in uppercase.

### Suspicious-term count

Counts occurrences of terms identified during feature engineering as potentially associated with suspicious job postings.

These features complement the NLP representation and provide additional information to the classifier.

---

# 🧮 Categorical Features

Categorical job attributes are transformed using the saved categorical encoder.

Examples include:

* Employment type
* Required experience
* Required education
* Industry
* Function
* Department

The encoder used during prediction is the same encoder saved during model development.

---

# 📊 Numerical Scaling

Numerical features are passed through the saved scaler before being combined with the other feature representations.

This ensures that numerical values are placed on the expected scale for the trained model.

---

# 🤖 Model

The project uses a **Linear SVM-based classification approach**.

The final feature representation combines:

```text
TF-IDF Features
        +
Categorical Features
        +
Numerical Features
        ↓
   Linear SVM
        ↓
Fraud / Legitimate
```

The model produces a decision score rather than simply returning a class label.

JobSentinel compares this score against the saved decision threshold.

```text
decision_score >= threshold
            ↓
       FRAUDULENT

decision_score < threshold
            ↓
       LEGITIMATE
```

---

# ⚠️ Risk Levels

The prediction system also provides a user-friendly risk interpretation.

Instead of exposing only the raw machine-learning output, the application presents the result as a risk assessment.

This allows users to understand the model's output more easily.

> **Important:** The risk score is a machine-learning assessment and should not be treated as a guaranteed determination that a job posting is fraudulent.

---

# 🔎 Explainable Risk Signals

JobSentinel also generates supporting signals to make the prediction easier to understand.

Examples include characteristics such as:

* Suspicious terminology
* Excessive contact information
* External URLs
* Unusual text patterns
* Missing job information
* Other engineered characteristics

These signals are intended to provide context around the prediction.

---

# 🌐 Adzuna API Integration

JobSentinel uses the **Adzuna Jobs API** to retrieve real job opportunities.

The application sends parameters such as:

```text
what     → Job title / search keywords
location → Requested location
```

For example:

```text
what=Python Developer
location=Bangalore
```

The API response is then normalized into the application's internal job structure.

---

# 🔄 Job Data Normalization

External API responses can contain inconsistent or optional fields.

JobSentinel normalizes these values before displaying them.

The normalized structure includes:

```text
title
company
location
salary_min
salary_max
salary_is_predicted
contract_type
contract_time
category
description
created
redirect_url
id
adref
```

This allows the Streamlit interface to work with a predictable structure regardless of the original API response.

---

# 🖥️ Application Workflow

## Step 1 — Enter Job Information

The user enters information about a job posting.

Examples:

```text
Job Title
Company
Description
Requirements
Benefits
Employment Type
Experience
Education
Industry
Function
Department
Salary
Location
```

---

## Step 2 — Analyze

The application prepares the posting and passes it through the trained machine-learning pipeline.

---

## Step 3 — Receive Fraud Assessment

The user receives:

* Prediction
* Risk level
* Decision score
* Threshold
* Risk explanation/signals

---

## Step 4 — Find Similar Jobs

The application uses the job title and location to search Adzuna.

---

## Step 5 — Display Opportunities

The user receives relevant job opportunities with information such as:

* Job title
* Company
* Location
* Salary
* Job category
* Employment type
* Description
* External application link

---

# 🛠️ Technology Stack

| Technology      | Purpose                            |
| --------------- | ---------------------------------- |
| Python          | Core programming language          |
| Pandas          | Data processing                    |
| NumPy           | Numerical operations               |
| Scikit-learn    | Machine learning and preprocessing |
| SciPy           | Sparse feature combination         |
| Joblib          | Model persistence                  |
| Streamlit       | Web application                    |
| Requests        | API communication                  |
| python-dotenv   | Local environment configuration    |
| Adzuna API      | Real-time job search               |
| Git             | Version control                    |
| GitHub          | Source code hosting                |
| Streamlit Cloud | Application deployment             |

---

# 📁 Project Structure

```text
JobSentinel/
│
├── app.py
│
├── src/
│   ├── predict.py
│   ├── job_recommender.py
│   ├── job_similarity.py
│   └── ...
│
├── models/
│   ├── best_model.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── categorical_encoder.joblib
│   ├── numeric_scaler.joblib
│   ├── feature_config.json
│   └── decision_threshold.txt
│
├── data/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── reports/
│   └── ...
│
├── tests/
│   └── ...
│
├── .gitignore
├── requirements.txt
└── README.md
```

> The exact contents of folders may evolve as the project develops.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/amanbaig77/JobSentinel.git
cd JobSentinel
```

---

## 2. Create a virtual environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a local `.env` file:

```env
ADZUNA_APP_ID=your_adzuna_app_id
ADZUNA_APP_KEY=your_adzuna_app_key
```

Never commit this file to GitHub.

Your `.gitignore` should contain:

```text
.env
.venv/
__pycache__/
*.pyc
```

---

# ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local Streamlit URL displayed in the terminal.

Usually:

```text
http://localhost:8501
```

---

# ☁️ Streamlit Cloud Deployment

The application can be deployed through Streamlit Cloud.

The required Adzuna credentials should be added through Streamlit Cloud Secrets.

Example:

```toml
ADZUNA_APP_ID = "your_app_id"
ADZUNA_APP_KEY = "your_app_key"
```

The secrets should **never** be committed to the GitHub repository.

---

# 🧪 Testing

The project can be tested locally by verifying the individual components.

### Test Adzuna credentials

```bash
python -c "from src.job_recommender import ADZUNA_APP_ID, ADZUNA_APP_KEY; print('APP_ID:', bool(ADZUNA_APP_ID)); print('APP_KEY:', bool(ADZUNA_APP_KEY))"
```

### Test Adzuna job search

```bash
python -c "from src.job_recommender import search_adzuna; print(search_adzuna({'title':'Python Developer','location':'Bangalore'}, 5))"
```

### Test Python syntax

```bash
python -m py_compile src/job_recommender.py
```

---

# 🧪 Example Search

Input:

```text
Job Title: Python Developer
Location: Bangalore
```

Example API result:

```text
Backend Developer- Python
Kyndryl
Bangalore, Karnataka
```

```text
Senior Python Full Stack Developer
RWS
Bangalore, Karnataka
```

```text
Python Developer
GrowthFalcons
Bangalore, Karnataka
```

The exact results can change because the application uses live job data from the external API.

---

# 🔒 Security Considerations

JobSentinel uses external API credentials.

The following practices should be followed:

* Never commit `.env`
* Never expose API keys in source code
* Use Streamlit Secrets for deployment
* Rotate exposed credentials
* Avoid sharing credentials in screenshots
* Keep private configuration outside Git

---

# ⚠️ Limitations

JobSentinel is a machine-learning based decision-support application and has several limitations.

### 1. Model limitations

The classifier's performance depends on the quality and characteristics of the training data.

### 2. False positives

A legitimate job may occasionally be classified as suspicious.

### 3. False negatives

A fraudulent job may occasionally appear legitimate.

### 4. External API dependency

Job recommendations depend on the availability and response of the Adzuna API.

### 5. Dynamic job data

Job listings, availability, descriptions, companies, and search results can change over time.

### 6. Risk score interpretation

The risk assessment should be treated as an indicator rather than definitive proof of fraud.

---

# 🚀 Future Improvements

Potential future improvements include:

* More advanced transformer-based NLP models
* Improved fraud-specific feature engineering
* Model calibration
* Better explainability using SHAP or similar techniques
* User-specific job recommendations
* Job bookmarking
* Job application tracking
* Personalized candidate profiles
* Resume-to-job matching
* Automated resume compatibility scoring
* More job APIs
* Advanced recommendation ranking
* Database-backed job history
* User authentication
* Analytics dashboard
* Automated model retraining
* Model monitoring
* Production-grade API backend

---

# 🎯 Project Goals

The main goals of JobSentinel are:

### For job seekers

Help users identify potentially suspicious job listings before spending time or sharing personal information.

### For job discovery

Help users find alternative opportunities related to a job they are investigating.

### For machine-learning development

Demonstrate a complete ML application workflow from:

```text
Data
 ↓
Feature Engineering
 ↓
Model Training
 ↓
Model Persistence
 ↓
Inference
 ↓
Risk Analysis
 ↓
External API Integration
 ↓
Web Application
 ↓
Cloud Deployment
```

---

# 💡 What This Project Demonstrates

JobSentinel demonstrates practical experience in:

* Machine Learning
* Classification
* NLP
* TF-IDF
* Feature Engineering
* Categorical Encoding
* Numerical Scaling
* Sparse Matrices
* Model Persistence
* API Integration
* Data Normalization
* Recommendation Systems
* Explainable Predictions
* Streamlit Development
* Environment Configuration
* Git/GitHub
* Cloud Deployment

---

# 📈 End-to-End Architecture

```text
                     JOBSENTINEL
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
      JOB POSTING INPUT         SEARCH PARAMETERS
             │                         │
             ▼                         ▼
     FEATURE ENGINEERING         ADZUNA API
             │                         │
             ▼                         ▼
        TF-IDF +                  RAW JOBS
       Categorical +                 │
        Numeric                      ▼
             │                  NORMALIZATION
             ▼                         │
      LINEAR SVM                      ▼
             │                  SIMILAR JOBS
             ▼
     FRAUD PREDICTION
             │
       ┌─────┴─────┐
       ▼           ▼
   LEGITIMATE   FRAUDULENT
       │           │
       └─────┬─────┘
             ▼
        RISK ANALYSIS
             │
             ▼
       STREAMLIT UI
             │
             ▼
          USER
```

---

# 🧑‍💻 Development Approach

The project was developed incrementally by integrating multiple components into one application:

1. Machine-learning prediction pipeline
2. Saved preprocessing objects
3. Fraud/risk interpretation
4. Streamlit interface
5. Adzuna API integration
6. Job normalization
7. Similar-job functionality
8. Environment configuration
9. Cloud deployment
10. Production testing

This approach demonstrates how a machine-learning model can be converted from an experimental pipeline into an end-user application.

---

# 📜 Disclaimer

JobSentinel is an educational and decision-support project.

The application does **not guarantee** that a job posting is fraudulent or legitimate.

Users should independently verify:

* Company identity
* Recruiter identity
* Official company website
* Email domain
* Job posting source
* Compensation claims
* Interview process
* Requests for money or sensitive information

Never send money or sensitive personal documents to an unknown recruiter solely because a job receives a low-risk prediction.

---

# ⭐ If You Find This Project Interesting

Feel free to explore the repository, test the application, and provide feedback.

---

# 👨‍💻 Author

**Aman Baig**

GitHub:
https://github.com/amanbaig77

Project:
https://github.com/amanbaig77/JobSentinel

---

## 📌 JobSentinel in One Sentence

> **JobSentinel is an end-to-end machine-learning application that analyzes job postings for potential fraud and helps users discover similar real-world job opportunities through live job-market data.**
