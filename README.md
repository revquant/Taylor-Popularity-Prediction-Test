# Taylor_Swift_Song_Popularity_Prediction
![Landing Page](https://github.com/arundhutiacad/Taylor_Swift_Song_Popularity_Prediction/blob/main/ALBUM%20COVERS.jpg)

Project Description: Predicting Taylor Swift's Popular Songs Using Various Machine Learning Models

Introduction:
Taylor Swift, a renowned singer-songwriter, has a vast discography with numerous chart-topping hits. In this project, we aim to build machine learning models to predict the popularity of Taylor Swift's songs based on certain features. The dataset consists of historical data for her songs, including attributes like lyrics, music genre, release year, album, and more. We will employ multiple machine learning algorithms to predict the popularity of her songs and identify which model performs the best in this task.

Objective:
The primary objective of this project is to use different machine learning algorithms to predict the popularity of Taylor Swift's songs based on the provided dataset. The models will be evaluated using relevant performance metrics, and the best-performing model will be identified.

Dataset:
The dataset includes a collection of Taylor Swift's songs, each represented by various features such as song lyrics sentiment analysis, genre, acousticness, danceability, instrumentalness, and more. Additionally, the dataset contains a popularity score for each song, derived from chart positions, streaming data, and other metrics.

Methodology:
1. Data Preprocessing:
   - Handle missing values, if any.
   - Convert categorical features into numerical representations using one-hot encoding or label encoding.
   - Scale numerical features to a similar range if required.

2. Feature Selection:
   - Identify the most relevant features that contribute significantly to the popularity prediction task.
   - Use feature importance techniques to select the most informative attributes.

3. Model Selection:
   - Implement five different machine learning models: Linear Regression, Support Vector Machine (SVM), XGBoost, Random Forest, Decision Tree, and Gradient Boosting.
   - Tune hyperparameters for each model using techniques like grid search or random search.

4. Model Training and Evaluation:
   - Split the dataset into training and testing sets.
   - Train each model on the training data.
   - Evaluate the models using appropriate metrics such as Mean Squared Error (MSE), R-squared (R2), and Mean Absolute Error (MAE) to measure prediction accuracy.

5. Model Comparison:
   - Compare the performance of all models based on their evaluation metrics.
   - Identify the best-performing model for predicting Taylor Swift's song popularity.

6. Prediction and Interpretation:
   - Use the best model to make predictions on new or unseen songs to estimate their popularity.
   - Interpret the results and analyze which features contribute most to a song's popularity.

Conclusion:
The project aims to build and compare multiple machine learning models to predict the popularity of Taylor Swift's songs based on various song attributes. The best-performing model will be selected, and insights will be drawn on the essential features that contribute to a song's success. This analysis can provide valuable information for songwriters, music producers, and the music industry in general to understand the factors that drive the popularity of songs.
