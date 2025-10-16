**🏦 AI Credit Default Prediction System**

An end-to-end Python project that predicts credit card default using machine learning.
Includes data preprocessing, model training, evaluation, and an interactive Streamlit app for users to predict default risk.

✨ Features

📊 Predicts credit card default using Random Forest.

🔄 Handles imbalanced data using SMOTE

🧮 Interactive Streamlit app for user-friendly predictions

📈 Shows probability of default and risk level (High/Low)

💾 Saves trained model and scaler for reuse

🛠️ How to Use
python -m streamlit run app.py

Enter customer data in the app to get predictions:

Credit limit, age, payment history, bill amounts, and payment amounts

Click Predict to see the probability of default and risk level

📁 File Structure
Credit-Default-Prediction-Project/
│
├── data/
│   └── default_of_credit_card_clients.xls      # Dataset
├── models/
│   ├── rf_credit_model.pkl                     # Saved Random Forest model
│   └── scaler.pkl                              # StandardScaler
├── train_model.py                              # Script to train and save model
├── app.py                                      # Streamlit app
└── README.md                                   # Project description

✅ Example

Streamlit Input Example:

Credit Limit: 50000
Age: 35
PAY_0: -1
BILL_AMT1: 30000
PAY_AMT1: 5000
...


Output:

Probability of Default: 23%
Risk Level: Low

🧑‍💻 Author

Built with 💙 by Jaypal Rathod

Feel free to copy, modify, and enhance for your own projects!
