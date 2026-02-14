# ML Assignment 2: Classification Models & Streamlit Deployment

## a. Problem Statement
The objective of this assignment is to develop a machine learning solution to classify breast cancer tumors as either **Malignant (M)** or **Benign (B)** based on a set of cell nuclei characteristics. This involves implementing six different classification algorithms, evaluating their performance using standard metrics, and deploying a user-friendly Streamlit web application to demonstrate the models' predictive capabilities. This tool aims to assist in early diagnosis by providing accurate predictions from bio-marker data.

## b. Dataset Description
- **Dataset Name**: Breast Cancer Wisconsin (Diagnostic) Data Set.
- **Source**: UCI Machine Learning Repository / Kaggle.
- **Features**: 30 numeric features representing characteristics of the cell nuclei present in the digitized image of a Fine Needle Aspirate (FNA) of a breast mass. Features include radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.
- **Target Variable**: `diagnosis` (M = Malignant, B = Benign).
- **Instance Count**: 569 instances.
- **Feature Count**: 30 features (plus ID and diagnosis).

## c. Models Used
The following six classification models were implemented and evaluated:
1.  **Logistic Regression**: A linear model using a logistic function to model a binary dependent variable.
2.  **Decision Tree Classifier**: A non-parametric supervised learning method used for classification.
3.  **k-Nearest Neighbor (kNN)**: A non-parametric method used for classification and regression.
4.  **Naive Bayes Classifier (Gaussian)**: A probabilistic classifier based on Bayes' theorem.
5.  **Random Forest (Ensemble)**: An ensemble learning method ensuring better predictive performance.
6.  **XGBoost (Ensemble)**: An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.

### Comparison Table
| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.9737** | **0.9974** | **0.9762** | **0.9535** | **0.9647** | **0.9439** |
| Decision Tree | 0.9474 | 0.9440 | 0.9302 | 0.9302 | 0.9302 | 0.8880 |
| kNN | 0.9474 | 0.9820 | 0.9302 | 0.9302 | 0.9302 | 0.8880 |
| Naive Bayes | 0.9649 | **0.9974** | 0.9756 | 0.9302 | 0.9524 | 0.9253 |
| Random Forest | 0.9649 | 0.9953 | 0.9756 | 0.9302 | 0.9524 | 0.9253 |
| XGBoost | 0.9561 | 0.9908 | 0.9524 | 0.9302 | 0.9412 | 0.9064 |

## d. Observations
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the **highest accuracy (97.37%)** and AUC score. It performed exceptionally well, likely due to the dataset's features being linearly separable to a high degree after standardization. |
| **Decision Tree** | Performed well (94.74%) but had the lowest AUC among the group. As a single tree, it might have slightly overfitted or lacked the robustness of ensemble methods. |
| **kNN** | Matched Decision Tree in accuracy (94.74%) but had a better AUC (0.9820), suggesting it's quite effective in ranking probabilities for this dataset. |
| **Naive Bayes** | Surprisingly strong performance (96.49%), tying with Random Forest. Its assumption of feature independence seems to hold reasonably well for these measurements. |
| **Random Forest** | excellent performance (96.49%) and very high AUC (0.9953). It provided a stable and high-confidence classification, reducing variance compared to the single Decision Tree. |
| **XGBoost** | Very strong performer (95.61%) with high AUC. While slightly behind Logistic Regression in raw accuracy on this specific test split, it remains a robust and powerful model. |

---
**Note**: All models achieved >94% Accuracy, making them all viable for this diagnostic task, with Logistic Regression and Random Forest leading the pack.
