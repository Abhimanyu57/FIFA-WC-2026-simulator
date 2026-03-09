# FIFA-WC-2026-simulator
Machine learning–based FIFA World Cup 2026 simulator that predicts match outcomes and simulates the entire tournament using probabilistic modeling and an interactive Streamlit interface.
This project builds a machine learning system that predicts international football match outcomes and simulates the entire FIFA World Cup 2026 tournament.

Using historical match data, the model estimates the probability of three possible outcomes:

Home team win

Draw

Away team win

An interactive web interface allows users to predict individual matches or run a full World Cup simulation from group stages to the final.

Project Overview

The goal of this project is to demonstrate a complete machine learning workflow applied to sports analytics. Historical international match results are used to derive team strength metrics and train a classification model capable of estimating match outcome probabilities.

These predictions are then used in a tournament simulation to model how the World Cup could unfold.

Features

• Predict match outcomes between qualified teams
• Probabilistic predictions (Win / Draw / Loss)
• Full World Cup simulation including group stage and knockout rounds
• Randomized tournament outcomes using probabilistic sampling
• Interactive web interface built with Streamlit

Machine Learning Approach

The model uses historical match results to calculate team performance metrics.

Two main features are derived:

Attack strength – average goals scored per match

Defense strength – average goals conceded per match

The difference between teams' attacking and defensive strengths is used as input to a classifier trained to estimate match outcome probabilities.

The model is implemented using XGBoost, which performs well on structured tabular datasets.

Technologies Used

Python
Pandas
NumPy
Scikit-learn
XGBoost
Streamlit

How It Works

Historical international match data is loaded and cleaned.

Team attacking and defensive statistics are calculated.

Features representing relative team strength are created.

A machine learning classifier is trained to predict match outcomes.

Predictions are used to simulate matches throughout the tournament.

The tournament progresses through group stage and knockout rounds until a champion is determined.

Running the Application

Install dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

The application will launch locally in your browser.

Dataset

The model is trained on historical international football match results. Friendly matches are excluded to reduce noise and focus on competitive fixtures.

Project Purpose

This project demonstrates several core machine learning concepts:

Data preprocessing

Feature engineering

Supervised classification

Probabilistic prediction

Monte Carlo style tournament simulation

Interactive model deployment

It serves as a learning project showing how sports data can be transformed into a predictive modeling problem.

Future Improvements

Potential extensions include:

Incorporating FIFA rankings or Elo ratings

Adding recent form statistics

Improving model calibration

Running large-scale Monte Carlo simulations to estimate tournament win probabilities
