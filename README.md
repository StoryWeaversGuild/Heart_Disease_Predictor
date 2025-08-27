## **❤️ Heart Disease Prediction Project**

This project predicts the likelihood of heart disease using machine learning. It establishes a **Logistic Regression** model as a baseline and then implements an advanced **Random Forest Classifier** with hyperparameters tuned via **Bayesian Optimization**, achieving a final test accuracy of **89%**.

### **📋 Table of Contents**

* [Project Overview](https://www.google.com/search?q=%23project-overview)  
* [Tech Stack](https://www.google.com/search?q=%23tech-stack-%EF%B8%8F)  
* [Project Structure](https://www.google.com/search?q=%23project-structure-)  
* [Installation](https://www.google.com/search?q=%23installation-%EF%B8%8F)  
* [Usage](https://www.google.com/search?q=%23usage-)  
* [Modeling Workflow & Results](https://www.google.com/search?q=%23modeling-workflow--results-)  
* [Future Improvements](https://www.google.com/search?q=%23future-improvements-)  
* [Contributing](https://www.google.com/search?q=%23contributing-)  
* [License](https://www.google.com/search?q=%23license)

### **📝 Project Overview**

This project aims to build a reliable predictive model for early-stage heart disease detection using patient data. The analysis follows a structured machine learning workflow, which includes:

1. **Data Preprocessing**: Cleaning and preparing the dataset by handling missing values and encoding categorical features.  
2. **Baseline Modeling**: Establishing a performance benchmark with a **Logistic Regression** model.  
3. **Advanced Modeling**: Implementing a **Random Forest** model and fine-tuning its hyperparameters using **Bayesian Optimization** to achieve higher accuracy.

The entire process is documented in the Heart\_Disease\_Prediction.ipynb notebook.

### **🛠️ Tech Stack**

* **Language**: Python 3.x  
* **Core Libraries**:  
  * scikit-learn (for modeling and preprocessing)  
  * pandas & numpy (for data manipulation)  
  * bayesian-optimization & skopt (for hyperparameter tuning)  
  * matplotlib & seaborn (for visualization)  
  * joblib (for saving/loading models)

### **📂 Project Structure**

.  
├── Heart\_Disease\_Prediction.ipynb   \# Main notebook with all code and analysis  
├── heart\_rf\_model\_tuned.pkl         \# Saved tuned Random Forest model  
├── heart\_rbf\_features\_logreg\_tuned.pkl \# Saved tuned Logistic Regression model  
├── heart\_scaler.pkl                 \# Saved StandardScaler object  
└── README.md                        \# Project documentation

### **⚙️ Installation**

To run this project locally, follow these steps:

1. **Clone the repository:**  
   git clone https://github.com/your-username/Heart\_Disease\_Predictor.git  
   cd Heart\_Disease\_Predictor

2. **Create a virtual environment (recommended):**  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`

3. Install the required dependencies:  
   (It is recommended to create a requirements.txt file for easier installation)  
   pip install scikit-learn pandas numpy bayesian-optimization scikit-optimize matplotlib seaborn jupyter

### **🚀 Usage**

1. **Start Jupyter Notebook:**  
   jupyter notebook

2. Open and run the notebook:  
   Navigate to Heart\_Disease\_Prediction.ipynb and execute the cells to see the full analysis, from data cleaning to model evaluation.

### **📊 Modeling Workflow & Results**

The project employed a two-stage modeling approach to ensure robust evaluation:

1. **Baseline Model (Logistic Regression)**: An initial model was trained to establish a performance baseline.  
   * **Accuracy**: \~84%  
2. **Advanced Model (Random Forest with Bayesian Optimization)**: A more complex model was implemented and tuned to find the optimal combination of n\_estimators, max\_depth, and min\_samples\_split.  
   * **Final Test Accuracy**: \~89%

The results clearly show that the tuned Random Forest model provides a significant improvement in predictive accuracy over the baseline.

### **📌 Future Improvements**

* **Experiment with Other Models**: Implement gradient boosting models like XGBoost or LightGBM to explore potential performance gains.  
* **Build a User Interface**: Develop a simple web application using Streamlit or Flask for interactive, user-friendly predictions.  
* **Deployment**: Containerize the final model using Docker and deploy it to a cloud service for broader accessibility.

### **🤝 Contributing**

Contributions are welcome\! Please fork the repository and submit a pull request with any enhancements.

### **📄 License**

This project is licensed under the MIT License.