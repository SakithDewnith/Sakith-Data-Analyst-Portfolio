# Sakith Fernando - Data Analyst / Data Science Portfolio

## Table of Contents

* [1. Customer Churn Risk & Revenue Exposure Analytics](#1-customer-churn-risk--revenue-exposure-analytics)
* [2. Chest X-Ray Pneumonia Detection using Deep Learning](#2-chest-x-ray-pneumonia-detection-using-deep-learning)


---

## 1. Customer Churn Risk & Revenue Exposure Analytics

**📋 README:** [View Documentation](https://github.com/SakithDewnith/Customer-churn-revenue-exposure/blob/main/README.md)

**🚀 Live Demo:** [Open Dashboard](https://sakith-churn-analysis-dashboard.streamlit.app/)

**📓 Data Analysis Notebook:** [View Notebook](https://github.com/SakithDewnith/Customer-churn-revenue-exposure/blob/main/churn%20analysis.ipynb)

<br>

**Goal:** 

Identify customer segments with the highest future revenue exposure, uncover the key drivers contributing to revenue risk, and prioritize retention opportunities based on potential business impact.

<br>

**Analytical Workflow:**
<p align="center">
  <img 
    width="524" 
    height="100" 
    alt="Gemini_Generated_Image_g3te9bg3te9bg3te" 
    src="https://github.com/user-attachments/assets/24c919d4-67e4-425a-9e02-b8e4f0eb60e8"
  />
</p>

<br>

**Description:**

This project analyzed a telecom customer dataset containing 7,043 customer records with demographic information, subscription details, contract types, payment methods, monthly charges, and churn status.

The analysis focused on identifying where future revenue risk is concentrated by applying historical churn behaviour to active customers. Using **SQL** and **Python**, customer data was analyzed to evaluate churn patterns, estimate future revenue exposure, identify high-impact customer segments, and uncover the main drivers contributing to revenue risk. A **Streamlit dashboard** was developed to present customer risk segments, exposure drivers, and retention insights.

<br>

**Skills:**

Customer Analytics | Customer Segmentation | Churn Analysis | Revenue Exposure Analysis  
Exploratory Data Analysis (EDA) | Business Analytics | Data Visualization | Dashboard Development

<br>

**Technology:**

SQL | Python | Pandas | NumPy | PostgreSQL | Streamlit | Plotly

<br>

**Results & Key Findings:**

- Identified **Month-to-month customers** as the highest future revenue exposure segment:
  - **42.71% churn rate**
  - **2,220 active customers**
  - **$58K+ projected monthly revenue exposure**

- Within this priority segment, identified the main revenue risk drivers:
  - Customers without **Tech Support** → **$93K+ projected monthly exposure**
  - Customers without **Online Security** → **$91K+ projected monthly exposure**

- Prioritized retention opportunities based on **potential revenue impact**, enabling targeted actions instead of applying the same strategy across all customers.


<br>

**Business Recommendations:**

- Prioritize retention strategies for high-exposure Month-to-month customers.
- Test contract upgrade offers to encourage longer-term customer commitment.
- Test Tech Support and Online Security adoption campaigns for customers with identified service gaps.
- Measure retention impact through controlled experiments and continuously monitor future revenue exposure trends.

<br>

**Dashboard Preview:**

<table align="center" cellspacing="20">
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/451d485e-4aed-4039-887a-a0647ff8bb95" width="400"/>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/57612e43-a7da-4662-9ed4-3a11e8a827ef" width="400"/>
    </td>
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
