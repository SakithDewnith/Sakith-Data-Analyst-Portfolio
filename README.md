# Sakith Fernando - Data Analyst / Data Science Portfolio

## Table of Contents

* [1. Customer Churn & Revenue Exposure Analytics](#1-customer-churn--revenue-exposure-analytics)
* [2. Chest X-Ray Pneumonia Detection using Deep Learning](#2-chest-x-ray-pneumonia-detection-using-deep-learning)
* [3. Employee Salary Analysis & Prediction ](#3-employee-salary-analysis--prediction)

---

## 1. Customer Churn & Revenue Exposure Analytics

**📋 README:** [View Documentation](https://github.com/SakithDewnith/Customer-churn-revenue-exposure/blob/main/README.md)

**🚀 Download the Power BI Dashboard (.pbix):** [Open Dashboard](https://github.com/SakithDewnith/Customer-churn-revenue-exposure/blob/main/Customer_Churn_Revenue_Exposure_Dashboard.pbix)

**📓 Data Analysis Notebook:** [View Notebook](https://github.com/SakithDewnith/Customer-churn-revenue-exposure/blob/main/churn%20analysis.ipynb)

<br>

**Goal:** 

Identify customer segments with elevated churn, estimate their potential monthly revenue exposure, and prioritize retention opportunities based on potential business impact.

**Analytical Workflow:**
<p align="center">
<img width="624" height="100" alt="Create_workflow_diagram_for_analysis" src="https://github.com/user-attachments/assets/4f6dbc6c-36c9-4795-8a46-e55e23a06726" />
</p>

<br>

**Description:**

This project analyzed **7,043 telecom customer records** to identify where customer churn and potential monthly revenue exposure are concentrated. Using **SQL (PostgreSQL)** and **Python (Pandas)**, I cleaned and validated the data, screened 17 customer attributes to select **Contract Type, Internet Service, and Payment Behaviour** for deeper analysis, and prioritized one customer segment within each factor. I then analyzed subgroups with elevated churn and ranked qualifying groups by estimated monthly revenue exposure to highlight potential retention opportunities. An interactive **Power BI dashboard** enables users to compare priority segments, explore churn patterns, and investigate the customer subgroups contributing to revenue exposure.

<br>

**Skills:**

- **SQL (PostgreSQL):** Data Exploration (EDA), Data Cleaning, Data Validation, Aggregation, Conditional Aggregation (FILTER), GROUP BY Analysis, Calculated Metrics

- **Python:** Pandas, Matplotlib, SciPy, scikit-learn, Cramér's V, Logistic Regression, Permutation Feature Importance, Customer Segmentation, Factor Screening, Churn Analysis, Revenue Exposure Estimation, Driver Analysis, Data Visualization

- **Power BI:** DAX Measures, Data Modeling, Interactive Filtering, KPI Cards, Data Visualization, Interactive Dashboards

<br>

**Results & Key Findings:**

- **Overall Churn:** Analyzed 7,043 customers and found a 26.54% historical churn rate, with approximately $139.1K in monthly charges associated with churned customers.

- **Priority Segments:** Screened 17 customer attributes and selected three areas for deeper analysis: Month-to-month contracts (42.71% churn; $58.3K estimated monthly exposure), Fiber optic internet (41.89%; $70.8K), and Electronic check payments (45.29%; $43.5K).

- **Key Driver Findings:** Identified the highest-exposure subgroups within each segment: customers without Tech Support for Month-to-month ($46.9K/month), customers using paperless billing within Fiber optic ($55.9K/month), and Fiber optic users paying by Electronic check ($36.6K/month). Customers without Online Security also appeared among the top three subgroups in all three segments.

- **Business Dashboard:** Developed an interactive Power BI dashboard to compare churn rates, explore priority segments, and investigate customer subgroups with elevated churn and estimated monthly revenue exposure.

**Business Recommendations:**

- Test longer-term contract incentives for Month-to-month customers, particularly those using Fiber optic internet.
- Test targeted Tech Support and Online Security offers to assess whether service adoption improves customer retention.
- Investigate Fiber optic customers paying by Electronic check, as this subgroup has the highest estimated monthly revenue exposure within the Electronic check segment.
- Measure results against a comparable control group and expand successful initiatives only when the retained revenue justifies the cost.

These are proposed actions based on observed churn patterns; their effectiveness has not yet been tested.

<br>

**Dashboard Images (PowerBI):**

<table>
  <tr>
    <th>Default</th>
    <th>Contract Analysis</th>
    <th>Internet Analysis</th>
    <th>Payment Analysis</th>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/734733b2-d4bb-48a6-9f6d-79fb03d87f1e" width="250" alt="Default" /></td>
    <td><img src="https://github.com/user-attachments/assets/92389d44-038a-4f60-9c57-b2b720d3646f" width="250" alt="Contract Analysis" /></td>
    <td><img src="https://github.com/user-attachments/assets/49f3ff85-d982-47f8-af55-c17f0ed49ea2" width="250" alt="Internet Analysis" /></td>
    <td><img src="https://github.com/user-attachments/assets/e261b0b8-2990-471e-92ec-854faa0877d7" width="250" alt="Payment Analysis" /></td>
  </tr>
</table>



</div>

<br>

---

## 2. Chest X-Ray Pneumonia Detection using Deep Learning

**📋 README:** [View Documentation](https://github.com/SakithDewnith/pneumonia-detection-app/blob/main/README.md)

**🚀 Live Demo:** [Open Dashboard](https://sakith-pneumonia-detection.streamlit.app/)

**📓 Data Analysis Notebook:** [View Notebook](https://github.com/SakithDewnith/pneumonia-detection-app/blob/main/notebooks/Pneumonia_CNN_Detection%20final.ipynb)

<br>

**Goal:** 

Develop a deep learning-based pneumonia screening system to classify chest X-ray images as Normal or Pneumonia, achieving reliable detection performance with emphasis on reducing missed pneumonia cases.

<br>

**Workflow:**
<p align="center">
<img width="524" height="100" alt="Gemini_Generated_Image_cj9rm1cj9rm1cj9r" src="https://github.com/user-attachments/assets/2c7ac3a3-dfab-49ba-8001-31fca1a1ed21" />
</p>

<br>

**Description:**

This project developed an end-to-end deep learning-based chest X-ray classification system using Convolutional Neural Networks (CNN) to identify pneumonia patterns from medical images.

The analysis used a balanced chest X-ray dataset combining NIH ChestX-ray14 and Kaggle Chest X-Ray Images (Pneumonia) datasets, containing labeled Normal and Pneumonia images for model training and evaluation. The workflow involved preparing and balancing image data, improving X-ray quality through preprocessing techniques, training a CNN classification model, evaluating performance using healthcare-focused metrics, and deploying the trained model through an interactive Streamlit application.

The system enables users to upload a chest X-ray image and receive a prediction of Normal or Pneumonia along with the model confidence score.

<br>

**Data Preprocessing:**

| Technique | Purpose |
|---|---|
| Image Resizing (224×224) | Standardized input dimensions for CNN training |
| Grayscale Conversion | Reduced image complexity while preserving important patterns |
| CLAHE Enhancement | Improved X-ray contrast and highlighted important features |
| Pixel Normalization | Scaled pixel values to improve model training stability |
| Data Augmentation | Increased dataset diversity and improved model generalization |

<br>

**Skills:**

Computer Vision | Deep Learning | Medical Image Analysis | Image Classification
Data Preprocessing | Predictive Analytics | Model Training | Model Evaluation | Model Deployment

<br>

**Technology:**

Python | TensorFlow | Keras | OpenCV | NumPy | Pandas | Matplotlib | Streamlit

<br>

**Results & Key Findings:**

Developed a CNN-based pneumonia detection model achieving:

- **85.1% accuracy** on unseen chest X-ray images  
- **90% recall**, prioritizing pneumonia detection and reducing missed cases  
- **93.4% AUC-ROC**, demonstrating strong separation between Normal and Pneumonia classes  

Key findings from model evaluation:

- CLAHE preprocessing improved X-ray image contrast and helped highlight important image features.  
- Data augmentation improved model robustness and reduced overfitting during training.  
- High recall was prioritized over accuracy due to the importance of minimizing missed pneumonia cases in healthcare screening.

<br>

**Future Improvements:**

- Apply transfer learning models such as ResNet50, EfficientNet, and DenseNet.
- Add Grad-CAM explainability for model interpretation.
- Validate performance using larger and more diverse datasets.

<br>

**Dashboard Preview:**

<p align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/59a0bc4f-c844-49a1-b210-40171edef990" width="550" alt="Screenshot 1"><br>
        <sub><b>State: idle</b></sub>
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/34161686-00b3-456b-839c-64a09ad0dc00" width="550" alt="Screenshot 2"><br>
        <sub><b>State: Pneumonia Positive</b></sub>
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/d77d4a51-eac7-40a9-b48d-1fa56965c5f0" width="550" alt="Screenshot 3"><br>
        <sub><b>State: Pneumonia Negative</b></sub>
      </td>
    </tr>
  </table>
<br>

---

## 3. Employee Salary Analysis & Prediction 

**📋 README:** [View Documentation](https://github.com/SakithDewnith/CS_Group_09_Salary_prediction/blob/main/README)

**📓 My Model – Linear Regression::** [View Notebook](https://github.com/SakithDewnith/CS_Group_09_Salary_prediction/blob/main/notebooks/Sakith_FC110558/Linear_regression.ipynb)

<br>

**Goal:** 

Analyze employee salary trends, explore factors associated with salary differences, and develop and evaluate machine learning models for salary prediction.
<br>

**Workflow:**
<p align="center">
  <img
    width="624" height="100"
    alt="Employee Salary Prediction Project Workflow"
    src="https://github.com/user-attachments/assets/daec31e5-26db-4c20-9b4f-c37dd5df3880"
  />
</p>

<br>

**Description:**

This group project analyzes employee salary data to explore salary patterns across age, gender, education level, job title, and years of experience. The project involves data cleaning, exploratory data analysis, machine learning model development, performance evaluation, and dashboard visualization. Multiple regression algorithms were evaluated to compare their salary prediction performance. **My contributions included developing and evaluating a Linear Regression model, identifying potential data inconsistencies, and recommending a systematic data-cleaning process to help the team identify and address data-quality issues consistently**. The findings and visualizations provide insights into salary patterns and model performance.  

<br>

**Technical Methods:**

| Technique                       | Purpose                                                                                                     |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Data Cleaning                   | Removed missing values and duplicate records, standardized education labels, and filtered low salary values |
| Exploratory Data Analysis (EDA) | Analyzed salary distributions and relationships with age, experience, education, gender, and job title      |
| Feature Preprocessing           | Applied one-hot encoding to categorical variables and standard scaling to numerical features                |
| Rare Category Grouping          | Grouped infrequent job titles and education categories into an "Other" category                             |
| Linear Regression Pipeline      | Integrated preprocessing and Linear Regression into a single scikit-learn pipeline                          |
| Cross-Validation                | Applied five-fold cross-validation to assess model performance across different data splits                 |
| Model Evaluation                | Evaluated predictions using R², RMSE, actual-versus-predicted plots, and residual analysis                  |

<br>

**Skills:**

Data Cleaning | Data Quality Assessment | Exploratory Data Analysis (EDA) | Salary Trend Analysis | Data Visualization | Feature Preprocessing | Linear Regression | Model Evaluation | Cross-Validation

<br>

**Technology:**

Python | Pandas | NumPy | Matplotlib | Seaborn | scikit-learn | Joblib| Streamlit

<br>

**Results & Key Findings:**

**My Model — Linear Regression**

* **Model Performance:** Achieved a test R² score of **0.806** and a five-fold cross-validation mean R² of **0.819**.
* **Prediction Error:** Recorded an RMSE of approximately **22,598** on the test set.
* **Model Evaluation:** Used actual-versus-predicted plots and residual analysis to examine prediction accuracy and errors.
* **Salary Analysis:** Explored salary distributions and relationships with age, years of experience, education level, gender, and job title.

**Overall Group Project**

* **Data Cleaning:** Reduced the dataset from **6,704 to 1,782 records** after removing missing values, duplicate rows, and low salary values.
* **Model Comparison:** Random Forest achieved the highest reported test R² score of **0.98** and the lowest RMSE of **6,849** among the evaluated models.
* **Key Conclusion:** Tree-based models, particularly Random Forest, showed stronger reported predictive performance than Linear Regression in the model comparison.

<br>

**Future Improvements:**

* **Feature Engineering:** Explore additional features and transformations to improve Linear Regression performance.
* **Ensemble Methods:** Combine Random Forest with other well-performing models to investigate potential performance improvements.
* **Hyperparameter Tuning:** Optimize parameters for models with lower predictive performance.
* **More Training Data:** Collect additional reliable training data to improve model stability and generalization.
* **Alternative Algorithms:** Experiment with algorithms such as LightGBM and CatBoost.
<br>

**Model Deployment Recommendations**

* **Model Selection:** Consider Random Forest for deployment after validating its performance.
* **Model Monitoring:** Monitor prediction accuracy and retrain the model as new data becomes available.

<br>

---
