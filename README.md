# Customer Sentiment Analysis using Machine Learning

## 📊 Project Overview

This project implements a comprehensive **Customer Sentiment Analysis** system using machine learning techniques to classify customer reviews across multiple e-commerce platforms. The analysis follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology, ensuring a structured and rigorous approach to data science.

## 🎯 Business Objectives

- **Predict Customer Sentiment**: Build accurate ML models to classify reviews as Positive, Neutral, or Negative
- **Improve Customer Retention**: Identify at-risk customers through sentiment analysis
- **Product Enhancement**: Surface product issues through sentiment patterns
- **Platform Optimization**: Compare sentiment across platforms to identify best practices

## 📋 Dataset

- **Source**: [Customer Sentiment Dataset](https://www.kaggle.com/datasets/kundanbedmutha/customer-sentiment-dataset)
- **Size**: 25,000 customer feedback records
- **Coverage**: Amazon, Flipkart, Meesho, Myntra, JioMart, Swiggy, and 15+ other major e-commerce platforms
- **Features**: Customer demographics, purchase details, review text, ratings, service metrics
- **Target**: Sentiment classification (Positive/Neutral/Negative)

## 🤖 Machine Learning Models

The project evaluates multiple algorithms with hyperparameter optimization:

1. **K-Nearest Neighbors (KNN)** - Instance-based learning with SMOTE for class imbalance
2. **Decision Tree** - Hierarchical decision rules with balanced class weights
3. **Logistic Regression** - Linear probabilistic model with L1 regularization
4. **Random Forest** - Ensemble method combining multiple decision trees
5. **XGBoost** - Gradient boosting with advanced regularization

## 🛠️ Technical Implementation

### Data Processing Pipeline
- **Text Feature Extraction**: TF-IDF vectorization (100 features, uni/bi-grams)
- **Categorical Encoding**: Label encoding for demographic and platform features
- **Numeric Scaling**: StandardScaler for rating and response time features
- **Class Imbalance Handling**: SMOTE oversampling and balanced class weights

### Model Evaluation
- **Stratified Train-Test Split**: 80/20 split maintaining class distribution
- **Cross-Validation**: 5-fold stratified CV for hyperparameter tuning
- **Metrics**: Accuracy, Precision, Recall, F1-Macro score
- **Success Criteria**: F1-Macro ≥ 0.85, Train-Test gap < 10%

## 📊 Key Findings

- **Perfect Model Performance**: All models achieved 100% accuracy on synthetic dataset
- **Strong Predictive Features**: Customer rating shows perfect correlation with sentiment
- **Class Balance**: Near-equal positive (39.9%) and negative (39.7%) sentiment distribution
- **Text Insights**: TF-IDF captured sentiment indicators like "amazing experience", "customer service"

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

### Quick Start
1. Clone the repository
2. Install dependencies
3. Place `Customer_Sentiment.csv` in the project directory
4. Open `FCase_Study2_BSIS4_TeamPapi.ipynb` in Jupyter
5. Run all cells sequentially

## 📈 Results Summary

| Model | F1-Macro (CV) | Accuracy (Test) | Key Strengths |
|-------|---------------|-----------------|---------------|
| KNN | 1.0000 | 1.0000 | Handles non-linear patterns |
| Decision Tree | 1.0000 | 1.0000 | Interpretable rules |
| Logistic Regression | 1.0000 | 1.0000 | Probabilistic output |
| Random Forest | 1.0000 | 1.0000 | Robust ensemble |
| XGBoost | 1.0000 | 1.0000 | Advanced boosting |

## 👥 Team

**Team Papi Analytics**
- **James Andrew Dorado** - BSIS 4B, Data Analyst
- **Jemmar V. Padernal** - BSIS 4B
- **Kristine Joy Casaquite** - BSIS 4A
- **Harvey Kim Solano** - BSIS 4A

**Course**: CIS - 221: Analytics Application  
**Institution**: [Your Institution Name]

## 📄 Methodology

This project strictly follows the **CRISP-DM** framework:
1. **Business Understanding** - Define objectives and success criteria
2. **Data Understanding** - Explore and assess data quality
3. **Data Preparation** - Clean, transform, and engineer features
4. **Modeling** - Train and tune multiple algorithms
5. **Evaluation** - Assess model performance and generalization
6. **Deployment** - Prepare for production implementation

## 🔒 License

This project is licensed under the GPL v3.0 License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- Dataset provided by Kundan Bedmutha on Kaggle
- CRISP-DM methodology guidance
- Scikit-learn and ML community resources

---

*This project demonstrates end-to-end machine learning workflow for sentiment analysis in e-commerce, suitable for academic study only*
