# Customer Churn Predictor (Telco)

An end-to-end **Machine Learning** project that predicts whether a telecom customer is likely to churn.  
Built with **Python**, **Scikit-learn**, and **Streamlit**, this project covers data preprocessing, model training, and interactive web app deployment.

---

## Project Overview
Customer churn is a major challenge for subscription-based businesses like telecom companies.  
This project uses historical customer data to:
- Predict the probability of churn.
- Provide insights into key factors driving churn.
- Help businesses take proactive measures to retain customers.

---

## Features
- End-to-end ML pipeline (data cleaning → model training → evaluation).
- Preprocessing of both numeric and categorical features.
- Interactive Streamlit web app for predictions.
- Probability-based output with visualisations.
- Easily deployable to **Streamlit Cloud**.

---

## Project Structure
churn-project/
├─ data/
│ └─ telco_churn.csv # Dataset
├─ notebooks/
│ └─ 01_explore.ipynb # EDA notebook
├─ src/
│ ├─ data_preprocessing.py # Data cleaning and preprocessing
│ ├─ train_model.py # Model training script
│ ├─ utils.py # Helper functions (optional)
├─ app/
│ └─ streamlit_app.py # Streamlit app
├─ models/
│ └─ churn_pipeline.pkl # Saved model
├─ requirements.txt # Python dependencies
└─ README.md # Project documentation

Dataset
We use the Telco Customer Churn Dataset from Kaggle.
It contains customer demographic and service information, along with a churn indicator.

Target variable:
Churn → Yes (1) or No (0)

Usage
1. Train the Model
bash
Copy
Edit
python src/train_model.py
This:

Loads and preprocesses the dataset.

Trains the ML model.

Saves the trained pipeline to models/churn_pipeline.pkl.

2. Run the Streamlit App
bash
Copy
Edit
streamlit run app/streamlit_app.py
This:

Loads the trained pipeline.

Provides a sidebar for entering customer details.

Displays churn probability, prediction result, and a simple visualisation.


Streamlit Interface
Features:

Sidebar sliders, dropdowns, and inputs for customer details.

Instant prediction upon button click.

Probability display with a bar chart.

Colour-coded churn warning.

Example screenshot:


Model Evaluation
Accuracy

Precision

Recall

F1-score

ROC AUC

We chose RandomForestClassifier for its balance of performance and interpretability.


Deployment
Deploy to Streamlit Community Cloud
Push your project to GitHub.

Go to Streamlit Cloud.

Click "New App" → Select your repo and app/streamlit_app.py.

Deploy!

🛠 Tech Stack
Python 3.9+

Pandas & NumPy – Data handling

Scikit-learn – Machine learning

Joblib – Model persistence

Streamlit – Web app

Matplotlib/Seaborn – Visualisations


Future Improvements
Add SHAP explainability for feature contributions.

Deploy to AWS/GCP for scalability.

Implement real-time prediction API.


License
This project is licensed under the MIT License.
Feel free to use and modify for your own purposes.


Author
Your Name – Data Analyst / Data Scientist
 GitHub: @Chirchirp
 Email: chirkiruiphero@gmail.com