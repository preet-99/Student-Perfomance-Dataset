# 🎓 Student Performance Prediction using Random Forest

This project implements a machine learning pipeline to predict student performance (exam scores) 
based on demographic, social, and academic features. The pipeline includes data preprocessing, 
feature scaling, one-hot encoding, and training a Random Forest Regressor. The model is evaluated 
using MSE, RMSE, MAE, and R² score. Feature importance analysis highlights which factors 
influence student performance the most. 



## 📂 Project Structure

```text
├── Student_Performance.csv   # Dataset file (input data)  
├── model.pkl                 # Trained Random Forest model (saved after training)  
├── pipeline.pkl              # Preprocessing pipeline (saved after training)  
├── input.csv                 # Test set used for inference  
├── output.csv                # Predictions with target values  
├── main.py                   # Main Python script (entry point)  
├── requirements.txt          # Project dependencies  
└── README.md                 # Project documentation  
```

## ⚙️ Features
```text
● Preprocessing pipeline:
  ● Handling missing values (imputation)
  ● Standard scaling for numerical features
  ● One-hot encoding for categorical features (gender, race, parental education, etc.)
  ● Train/test split with stratification
● Random Forest Regressor for predicting student scores
● Model persistence using joblib
● Evaluation metrics:
  ● MSE (Mean Squared Error)
  ● RMSE (Root Mean Squared Error)
  ● MAE (Mean Absolute Error)
  ● R² score (Coefficient of Determination)
● Feature importance visualization (which factors affect performance most)
```

 ## 📊 Example Insights
```text
● R² score shows how well the model explains student performance.

● Feature importance highlights which attributes (study hours, parental education, etc.) contribute most to student outcomes.
```


# 👨‍💻 Authors

- @preet-99 - Preet Vishwakarma

## 🛠️ Installation
```bash 
git clone https://github.com/preet-99/Student-Perfomance-Dataset.git
cd student-performance-prediction
pip install -r requirements.txt
```