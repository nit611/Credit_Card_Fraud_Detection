# Credit Card Fraud Detection

Welcome to the project! This is an iteration of a creditcard fraud detection using Machine Learning. The dataset was obtained from <a  href='https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud'> this Kaggle dataset </a> by the Machine Learning Group of ULB. Do follow the link to get redirected to the Kaggle Data page, and of course, the several notebooks that have also worked on this project. <br>
The Jupyter notebook with python code, as well as the HTML version of it contains explanations to each and every part of the project! Make sure to go through all of it ! 

## Introduction

The dataset contains 28 anonymized variables, named V1, V2... V28, a Time variable indicating seconds elapsed since the first transaction in the data, an Amount variable whihc indicates the transaction amount of that observation, the target variable Class, which is a binary variable where 1 = Fraud and 0 = Legitimate Transaction.
The Kaggle introduction mentions that the anonymized 28 variables are obtained **after** a PCA transformation for data privacy protection. <br>


## Methodology

The project involves from exploration into the Amount variable, which gives a lot of insights into the transaction amounts. A correlation matrix is provided to show the mutual dependencies between variables in the data.<br>
One of the most important aspects of the project is that it is a classification problem on an **imbalanced datset**. The next major aspect of the project is that I do not resort to undersampling and oversampling techniques to deal with the severe class imbalance of the data.<br>
Legitimate transactions - 99.83% of the data,
Fraudulent transactions - 0.17% of the data.
After basic data cleaning, there are only 473 frauds, as opposed to 235,000+ legitimate transactions. 

#### No Undersampling or Oversampling

Oversampling or undersampling lets the machine learn from equal amounts of data for each class, enabling it to predict better on unseen data and generalize well. However, oversampling increases the minority class by an extent which creates non-existent patterns in the data, and creates way too many fake transactions. Similarly, the removal of majority of the data in the majority class all the way down to 473 observations remove well-needed information contained in the data to train. Consequently, there will be a generalization problem with the model. 

#### Class weight hyperparameter

Ensemble techniques such as Random Forest Classifier, Boosting algorithms, the Decision Tree, and also algorithms such as SVM and Logistic Regression have a parameter to adjust for the class weights in order to calculate their cost functions. It helps penalize mistakes while predicting the Fraud class while relatively reducing the weight to the majority class.
Changing and adjusting for the class weights and how they affect the performance of the model is all clearly explained the notebook above attached in the code section.
#### Financial Cost
I introduce a financial cost  - to not only calculate the metrics based on the machine learning evaluation results, but also a sense of how much a business can save with the model by calculating the costs involved in misclassifying a fraud and misclassifying a legitimate transaction. The explanations for this financial cost calculation based on the Confusion Matrix (which is a summary of True Positives, True Negatives, False Positives and False Negatives) is given in the notebook in the code!

## Main Findings
Experimenting on the Random Forest algorithm with different Class Weights combinations for legit transactions and fraud transactions gives us a picture of how the model's performance changes with changes in the weights.
The precision vs. recall for the model's performances also depended on the weights of the classes in Random Forest algorithm.
As you will see in the notebook, the precision is terrible when the recall is highest, but even for the slightest sacrifice of recall, the precision improves greatly. <br>
The financial cost involved in the model evaluation is calculated as: 
_cost_ = 0 x _TN_ + A x _TP_ + A x _FP_ + Amt x _FN_,
where A is the admin cost, and Amt is the transaction amount average for when the transaction is a fraud. The admin cost can be the cost involved in raising an alert in case the transaction is detected as fraud, and the resources used to process the alerts. Amt is the average transaction amount for all fraudulent transactions in the dataset. This is of course, for simplicity. 
Class Weights for Fraudulent Transactions vs. Model's Financial Savings

![The Class Weights vs. Model's Financial Savings](https://raw.githubusercontent.com/nit611/Credit_Card_Fraud_Detection/da_real_nit/output/weights_vs_cost.png)

##  Last Notes<br>

This is in no way the greatest model in real life, but a strong showcase on domain-specific design of ML models. <br>

Some takeaways for future iterations:<br>

* Only one algorithm is tried, where XGBoost and Neural Networks can work better at predicting frauds in such a heavily imbalanced dataset.

* Not blaming data privacy issues, some more feature extraction can be looked into.

* Coming up with reasonably accurate weighting for different classes can be useful for the future.

* KFold Cross validated Train-Test Split can be applied. Despite the 70% data taken for training would contain the nuances and distributional idiosyncrasies of the different variables in the X-matrix, there is always room to improve the model selection process, which will help the algorithm look at an accurate representation of the entire dataset - to not overfit on training data, and generalize well on the testing data.<br>
