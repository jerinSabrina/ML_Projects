# Churn prediction project

We start a new project: predicting customer churn for a telecom company.
In this first unit we look at the problem statement, how churn prediction
fits into machine learning as binary classification, and the dataset we
will work with for the whole module.

## The problem of churn

Churn is when a customer stops using a service. A telecom company has
many customers, and every month some of them decide to terminate their
contracts and switch to another provider. Losing customers is expensive,
so the company wants to identify the people who are likely to leave
before they actually do.

The idea is simple: for every customer, we compute a score between 0
and 1 - the probability that this customer is about to churn. Then the
company can target the customers with high scores and try to retain
them, for example by sending an email with discounts or special
promotions.

## The dataset

For this project we use the
[Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
from Kaggle. It contains information about customers of a telecom
company and whether they churned.
Kaggle dataset overview: 
(image/2_churn_data.png)

Each row is one customer. The columns describe:
The dataset : (image/2_churn_data_kaggle.png)
- Customer ID
- Demographics: gender, whether the customer is a senior citizen,
  whether they have a partner and dependents
- Services: phone service, multiple lines, internet service, online
  security, online backup, device protection, tech support, streaming
  TV and streaming movies
- Account information: contract type, paperless billing, payment method
- Charges: monthly and total charges
- Tenure: the number of months the customer has stayed with the company
- Churn: the target - whether the customer left

The whole module walks through this project step by step: we prepare
the data, set up a validation framework, do EDA and feature importance
analysis, then train a logistic regression model and use it to score
customers.

