# -FUTURE_ML_03
# 🤖 ProScreen AI: Enterprise ML Resume Screening System

An advanced, end-to-end Machine Learning web application designed to automate and optimize the technical recruitment process. Built entirely in Python using **Streamlit**, **Scikit-Learn**, and **spaCy**, this platform significantly reduces the time-to-hire by instantly evaluating, categorizing, and ranking candidate resumes against specific Job Descriptions.

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-orange)

---

## 🌟 Core Offerings & Features

This system transcends traditional keyword-matching ATS (Applicant Tracking Systems) by leveraging semantic AI to understand the _context_ of a candidate's experience.

### 1. 🎯 Intelligent Candidate Ranking

- **Composite Scoring Logic**: Candidates are objectively scored using a 50/50 hybrid algorithm.
  - **50% Semantic Similarity**: Uses a trained **TF-IDF Vectorizer** to calculate the cosine similarity between the Job Description and the Resume, capturing contextual alignment even when exact keywords differ.
  - **50% Exact Skill Matching**: Employs **spaCy's PhraseMatcher** for rapid, precise identification of hard technical skills.
- **Skill Gap Analysis**: Automatically highlights critical skills requested in the Job Description that are absent from the candidate's resume, allowing recruiters to ask targeted interview questions.

### 2. 🧠 Predictive Machine Learning Classification

- The engine features a trained **Random Forest Classifier** that predicts the most likely job category for any given resume (e.g., Data Scientist, Backend Engineer, HR Manager).
- This allows for automated talent pooling—even if a candidate applies for the wrong role, the system knows where they actually belong.

### 3. 📄 Universal Universal Document Parsing

- Seamlessly extracts text from standard formats (`.pdf`, `.docx`, `.txt`).
- **LinkedIn Integration**: Features custom heuristic-based parsing explicitly designed to understand and structure data exported directly from LinkedIn Profiles (PDFs).

### 4. 📊 Enterprise Reporting

- Features an integrated reporting module (`scripts/reporting.py`) that physicalizes the digital analysis into professional, downloadable PDF Summary Reports for hiring managers.
- Visualizes the competitive landscape via custom Seaborn/Matplotlib bar charts.

---

## 🏗️ System Architecture & Repository Structure

The project has been architected for scalability, separating the frontend UI from the ML inference engine, and isolating the training pipeline from production.

```text
resume_screening_system/
├── LICENSE                                  # MIT License definitions
├── README.md                                # This document
├── requirements.txt                         # Immutable environment dependencies
├── app.py                                   # Entry point: Streamlit Web UI
├── nlp_engine.py                            # Core backend: NLP parsing & ML Inference
│
├── models/                                  # Trained ML Artifacts (Do not modify)
│   ├── resume_classifier_v2.pkl             # Trained Random Forest
│   ├── tfidf_vectorizer_v2.pkl              # TF-IDF feature extractor
│   ├── label_encoder_v2.pkl                 # Target category encoder
│   └── skills.json                          # Base technical skill dictionary
│
├── data/                                    # Sample PDF resumes for local testing
├── datasets/                                # Raw & extracted datasets used for training
│   ├── Resume/                              # Extracted CSV dataset
│   └── dataset.zip                          # Original archived Kaggle dataset
│
├── scripts/                                 # MLOps, utility, and reporting logic
│   ├── train_v2.py                          # Pipeline to train & export the RF model
│   ├── reporting.py                         # Generates the downloadable PDF reports
│   └── demo_result.py                       # CLI demo script
│
├── tests/                                   # Validation and debugging scripts
│   ├── repro_issue.py                       # Issue reproduction templates
│   ├── test_advanced.py                     # Integration tests
│   └── verify_task3.py                      # Task validation script
│
└── outputs/                                 # Generated assets (Reports, charts)
```

---

## ⚙️ Model Details & Training Pipeline

The machine learning models driving this application (`models/*.pkl`) were trained using the `scripts/train_v2.py` pipeline.

1. **Dataset**: Trained on a diverse Kaggle dataset of categorized resumes.
2. **Text Preprocessing**: Resumes undergo aggressive cleaning via Regex (removing URLs, non-ASCII characters, punctuation, and converting to lowercase).
3. **Feature Extraction**: Uses `TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=5000, ngram_range=(1, 2))` to capture massive N-Gram context while heavily penalizing common stop words.
4. **Algorithm**: `RandomForestClassifier(n_estimators=100, n_jobs=-1)`. Chosen for its robustness against overfitting on sparse text matrices and its ability to output prediction confidence probabilities.

---

## 🚀 Quick Start: Run the System Locally

Follow these steps to deploy the application on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/resume_screening_system.git
cd resume_screening_system
```

### 2. Configure the Environment

It is strictly recommended to run this application within an isolated Virtual Environment.

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies & NLP Assets

Install the standard Python libraries and download the required spaCy English language model.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Launch the Streamlit Interface

Execute the following command from the root directory to start the local web server.

```bash
streamlit run app.py
```

_The application should automatically open in your default browser at `http://localhost:8501`._

---

## 🛠️ Retraining the Machine Learning Engine

If you acquire new resume data and wish to improve the classifier's accuracy or add new job categories:

1. Format your new dataset to match the structure of `datasets/Resume/Resume.csv`.
2. Execute the training pipeline:
   ```bash
   python scripts/train_v2.py
   ```
3. The script will automatically preprocess the data, retrain the Random Forest and TF-IDF models, output the new testing accuracy to the terminal, and silently overwrite the `.pkl` files in the `models/` directory.
4. Restart `app.py` to utilize the upgraded models.

---

## 🤝 Contribution Guidelines

We welcome contributions from the Open Source community!

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License & Legal

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for comprehensive details. Feel free to use, modify, and distribute this software within your own enterprise applications.

---

<p align="center">
  <i>Bringing intelligent automation to Talent Acquisition.</i>
</p>
