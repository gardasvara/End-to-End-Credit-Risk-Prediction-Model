# Credit Risk Prediction - ID/X Partners Virtual Internship

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)

## 📌 Project Overview
This project is part of the **ID/X Partners Data Scientist Virtual Internship Program** at Rakamin Academy. The main objective is to build an end-to-end Machine Learning solution to predict **Credit Risk** (Loan Default).

By analyzing historical lending data (2007-2014), we developed a model to classify whether a borrower is likely to repay their loan (Good Loan) or default (Bad Loan). This solution aims to help lending companies minimize financial losses by identifying high-risk applicants early.

## 📂 Dataset
The dataset contains historical loan data from Lending Club (2007-2014).
* **Source:** ID/X Partners (Rakamin Academy)
* **Rows:** 466,285 loan records
* **Target Variable:** `bad_loan` (Derived from `loan_status`)
    * `0`: Good Loan (Fully Paid, Current, In Grace Period)
    * `1`: Bad Loan (Charged Off, Default, Does not meet credit policy)

## 🛠️ Methodology

### 1. Data Understanding & Cleaning
* **Target Definition:** Created a binary target variable where `Charged Off` and `Default` statuses are considered as **1 (Bad Loan)**.
* **Handling Missing Values:** Dropped columns with >50% missing data. Imputed remaining numerical columns with Median and categorical columns with Mode.
* **Removing Data Leakage:** Removed features that are not available at the time of loan application (e.g., `total_pymnt`, `recoveries`, `last_pymnt_d`).

### 2. Feature Engineering
* **Term:** Converted "36 months" / "60 months" to numerical `36` / `60`.
* **Employment Length:** Converted "10+ years" to numerical `10`.
* **One-Hot Encoding:** Applied to categorical features like `home_ownership`, `verification_status`, and `purpose`.

### 3. Modeling Strategy
We experimented with multiple algorithms to find the best balance between ROC-AUC and computational efficiency:
1.  **Logistic Regression** (Baseline)
2.  **Decision Tree Classifier**
3.  **Gradient Boosting Classifier** (Final Model)

We used **StandardScaler** for feature scaling and handled class imbalance using adjusted probability thresholds.

## 📊 Model Performance
The final model (**Gradient Boosting**) achieved the best performance:

| Metric | Score | Note |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.7232** | Indicates good discriminative ability |
| **Accuracy** | 91% | At high probability threshold |
| **Recall (Bad Loan)** | Tunable | Adjusted via threshold (e.g., 0.15) to capture more defaults |

### Key Business Insights (Feature Importance)
Based on the model, the top factors influencing credit risk are:
1.  **Interest Rate (`int_rate`):** Higher interest rates strongly correlate with higher default risk.
2.  **Loan Term (`term`):** Longer-term loans (60 months) carry higher risk than short-term ones.
3.  **Debt-to-Income Ratio (`dti`):** Borrowers with higher debt loads relative to income are riskier.

## 💻 Tech Stack
* **Language:** Python
* **Libraries:**
    * `pandas`, `numpy` (Data Manipulation)
    * `matplotlib`, `seaborn` (Visualization)
    * `scikit-learn` (Modeling & Evaluation)

## 🚀 How to Run
1.  Clone this repository:
    ```bash
    git clone [https://github.com/username/repo-name.git](https://github.com/username/repo-name.git)
    ```
2.  Install required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3.  Run the Jupyter Notebook:
    ```bash
    jupyter notebook IDX_DS_Code.ipynb
    ```
    Or run the Python script directly:
    ```bash
    python IDX_DS_Code.py
    ```

## 👤 Author
**Gardasvara Mistortoify**
* ID/X Partners Data Scientist 
