# Car price prediction project
We build a model that predicts the price
of a car. In this first unit we look at the scenario, the dataset we will use,
the plan for the project, and where the code and the data live.

## The scenario

A user wants to sell a car on an
online classified website, and the site asks them to enter a price. If they
don't know how much the car is worth, they have to guess. We want to help:
The user describes the car, and our model suggests the best price.

## The dataset
This project is about the creation of a model for helping users to predict car prices. The dataset was obtained from [this 
kaggle competition](https://www.kaggle.com/CooperUnion/cardataset).

To train a model, we need data, and we use a dataset with car prices from
Kaggle. It has different characteristics of cars — the make (the
manufacturer), the model, the year, the engine type, the fuel type and other
properties.

![Car Features and MSRP dataset on Kaggle](image/kaggle_car_data.png)

The data explorer on Kaggle shows what is inside. Every row is one car, and
every column is one characteristic of that car.

![Kaggle data explorer showing the car dataset columns](image/kaggle_car_data_2.png)

One column is especially interesting for us: MSRP. It stands for
Manufacturer Suggested Retail Price — in other words, the price of a car.
This is exactly what we want to predict. The plan is to use all the other
features — the make, the model, the year, the engine and so on — to predict
this price.

## The project plan

We will do the project in several steps:

- Get the data and do exploratory data analysis: just look at the data and
  try to learn more about it.
- Prepare the dataset and train a linear regression model on it to predict
  the price of a car.
- Go into the details of how linear regression is implemented — we will
  actually implement it ourselves.
- Evaluate the quality of the model with RMSE. RMSE stands for root mean
  squared error, a metric for evaluating the quality of model predictions.
- Do some feature engineering: the process of creating new features, new
  characteristics that we can use for our model.
- Deal with numerical stability problems and see how to solve them with
  regularization.
- Finally, use the model.
