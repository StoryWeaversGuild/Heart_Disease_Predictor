# **❤️ Heart Disease Prediction Project**

This project focuses on predicting the likelihood of heart disease using various machine learning models. It includes data preprocessing, feature engineering, hyperparameter tuning, and model evaluation, with a final accuracy of up to 89%.

## **📋 Table of Contents**

* [Project Overview](https://www.google.com/search?q=%23project-overview)  
* [Key Features](https://www.google.com/search?q=%23key-features)  
* [Tech Stack](https://www.google.com/search?q=%23tech-stack-%EF%B8%8F)  
* [Project Structure](https://www.google.com/search?q=%23project-structure-)  
* [Installation](https://www.google.com/search?q=%23installation-%EF%B8%8F)  
* [Usage](https://www.google.com/search?q=%23usage-)  
* [Modeling & Results](https://www.google.com/search?q=%23modeling--results-)  
* [Future Improvements](https://www.google.com/search?q=%23future-improvements-)  
* [Contributing](https://www.google.com/search?q=%23contributing-)  
* [License](https://www.google.com/search?q=%23license)

## **📝 Project Overview**

Heart disease remains a critical global health issue. This project leverages machine learning to build predictive models that can identify the probability of a patient having heart disease based on a set of medical attributes. The primary goal is to create an accurate and reliable tool that could potentially assist in early diagnosis. The main Jupyter Notebook provides a complete, step-by-step workflow from data exploration to final model evaluation.

## **✨ Key Features**

* **Data Preprocessing**: Comprehensive steps for cleaning, scaling, and preparing the dataset.  
* **Multiple Models**: Implements and evaluates several classifiers, including Random Forest and Logistic Regression with Kernel Approximation.  
* **Advanced Tuning**: Utilizes **Bayesian Optimization** for efficient and effective hyperparameter tuning.  
* **Complete Workflow**: A single Jupyter Notebook (Heart\_Disease\_Prediction.ipynb) contains the entire analysis for clarity and reproducibility.  
* **Pre-trained Models**: Includes saved, pre-trained model files (.pkl) for immediate use.

## **🛠️ Tech Stack**

* **Language**: Python 3.x  
* **Core Libraries**:  
  * scikit-learn (for machine learning models and pipelines)  
  * pandas & numpy (for data manipulation and numerical operations)  
  * bayes-opt / skopt (for Bayesian optimization)  
  * matplotlib & seaborn (for data visualization)  
  * joblib (for saving and loading models)

## **📂 Project Structure**

.  
├── Heart\_Disease\_Prediction.ipynb   \# Main notebook with all code and analysis  
├── heart\_rf\_model.pkl               \# Saved baseline Random Forest model  
├── heart\_rf\_model\_tuned.pkl         \# Saved tuned Random Forest model  
├── heart\_rbf\_features\_logreg\_tuned.pkl \# Saved tuned Logistic Regression model  
├── heart\_scaler.pkl                 \# Saved StandardScaler object  
├── user\_testing\_dataset.csv         \# Sample dataset for testing  
└── README.md                        \# Project documentation

## **⚙️ Installation**

To get a local copy up and running, follow these simple steps.

1. **Clone the repository:**  
   git clone https://github.com/StoryWeaversGuild/Heart\_Disease\_Predictor.git  
   cd Heart\_Disease\_Predictor

2. **Create a virtual environment (recommended):**  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`

3. Install the required dependencies:  
   (It is highly recommended to create a requirements.txt file)  
   pip install scikit-learn pandas numpy bayes-opt scikit-optimize matplotlib seaborn jupyter

## **🚀 Usage**

After setting up the environment, you can explore the project:

1. **Start Jupyter Notebook:**  
   jupyter notebook

2. Open the notebook:  
   In your browser, navigate to and open the Heart\_Disease\_Prediction.ipynb file.  
3. Run the cells:  
   Execute the cells sequentially to follow the analysis, from data loading to model training, tuning, and evaluation.

## **📊 Modeling & Results**

Two primary models were developed and evaluated. The Random Forest classifier achieved the highest accuracy.

| Model | Accuracy | Key Technique |
| :---- | :---- | :---- |
| **Random Forest** | **\~89%** | Bayesian Optimization |
| **Logistic Regression \+ Kernel Approximation** | \~85% | Bayesian Optimization (skopt) |

A confusion matrix was also generated to provide a more detailed view of the classification performance, showing true positives, true negatives, false positives, and false negatives.

## **📌 Future Improvements**

* **Develop a Prediction Script**: Create a simple Python script that loads the saved models (.pkl files) and makes predictions on new, unseen data.  
* **Build a User Interface**: Develop an interactive web application using Streamlit or Flask to allow users to input their data and get a prediction.  
* **Deployment**: Containerize the application using Docker and deploy it to a cloud platform (e.g., AWS, Heroku, GCP) for public access.  
* **Experiment with Other Models**: Implement and evaluate other powerful algorithms like XGBoost, LightGBM, or simple neural networks.

## **🤝 Contributing**

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project  
2. Create your Feature Branch (git checkout \-b feature/AmazingFeature)  
3. Commit your Changes (git commit \-m 'Add some AmazingFeature')  
4. Push to the Branch (git push origin feature/AmazingFeature)  
5. Open a Pull Request

## **📄 License**

Distributed under the MIT License. See LICENSE file for more information.
