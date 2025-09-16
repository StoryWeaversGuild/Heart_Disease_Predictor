Heart Disease Prediction System
This README file outlines the complete workflow of the Heart Disease Prediction System, from data collection and analysis to model training, tuning, and finally, deployment as a web application using Flask.
📜 Table of Contents
Data Collection
Exploratory Data Analysis (EDA)
Data Preprocessing & Feature Scaling
Baseline Model: Logistic Regression
Evaluating Multiple ML Algorithms
Hyperparameter Tuning with K-Fold Cross-Validation
Deployment with Flask
Frontend Interface
1. Data Collection 💾
The dataset for this project was sourced from Kaggle. Kaggle provides a vast repository of real-world data, which is ideal for building and testing machine learning models.
Source: Kaggle Datasets
Method: The data was downloaded as a .csv file and loaded into our project environment for analysis.
2. Exploratory Data Analysis (EDA) 📊
A thorough EDA was performed to understand the patterns, relationships, and anomalies in the data.
Tool: We primarily used the Pandas library for data manipulation, and Matplotlib/Seaborn for visualization.
<br>
Visualizations: A correlation heatmap was generated to understand the relationships between different features.
Fig 1: Correlation between different features.
3. Data Preprocessing & Feature Scaling ⚙️
To prepare the data for our models, we performed several preprocessing steps, including one-hot encoding for categorical variables and feature scaling.
Feature Scaling: We used StandardScaler from Scikit-learn to standardize the features, ensuring they have a mean of 0 and a standard deviation of 1.
<br>
4. Baseline Model: Logistic Regression 🎯
A simple baseline model was established using Logistic Regression to set a performance benchmark.
Result: The baseline model achieved an accuracy of 84% on the test set.
5. Evaluating Multiple ML Algorithms 📈
We trained and evaluated several ML algorithms to find the best-performing model for our dataset.
Algorithms Tested: Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and XGBoost.
Performance Comparison:
Fig 2: Bar chart comparing the accuracy of different models.
6. Hyperparameter Tuning with K-Fold Cross-Validation 🎛️
The best model, Random Forest, was fine-tuned using K-Fold Cross-Validation to maximize its performance.
Final Result: After tuning, the Random Forest Classifier achieved an improved accuracy of 89.1%.
Final Model Evaluation: A confusion matrix was used to assess the final model's performance in detail.
Fig 3: Confusion matrix for the final Random Forest model.
7. Deployment with Flask 🚀
The final model was deployed as a web application using the Flask framework.
Framework: Flask was used to build the backend API that serves the model's predictions.
<br>
8. Frontend Interface 💻
A simple user interface was created to interact with the deployed model.
Technologies: HTML, CSS, and JavaScript
