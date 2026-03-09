# FIFA-WC-2026-simulator
Machine learning–based FIFA World Cup 2026 simulator that predicts match outcomes and simulates the entire tournament using probabilistic modeling and an interactive Streamlit interface.
This project builds a machine learning system that predicts international football match outcomes and simulates the entire FIFA World Cup 2026 tournament.

# FIFA World Cup 2026 Predictor & Tournament Simulator

A machine learning project that predicts international football match outcomes and simulates the entire FIFA World Cup 2026 tournament.

The system estimates the probability of three possible match outcomes:

- Home Team Win
- Draw
- Away Team Win

Users can also simulate the full tournament from the group stage to the final using an interactive web interface.

## Project Overview

This project demonstrates a complete machine learning workflow applied to sports analytics. Historical international match results are used to derive team strength metrics and train a classification model capable of predicting match outcomes.

These predictions are then used to simulate the progression of a World Cup tournament.


## Features

- Match outcome prediction between qualified teams
- Probabilistic predictions (Win / Draw / Loss)
- Full World Cup tournament simulation
- Group stage and knockout rounds
- Randomized tournament outcomes using probabilistic sampling
- Interactive interface built with Streamlit


## Machine Learning Approach

The model uses historical international match data to estimate team strength.

Two main features are derived:

- **Attack Strength** – average goals scored per match
- **Defense Strength** – average goals conceded per match

The difference between teams' attacking and defensive strengths is used as input features for a classification model.

The model is implemented using **XGBoost**, a gradient boosting algorithm well suited for structured tabular data.


## Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  


## How It Works

1. Historical international match data is loaded and cleaned.
2. Team attacking and defensive statistics are calculated.
3. Feature differences between teams are computed.
4. A classification model is trained to predict match outcomes.
5. Predictions are used to simulate tournament matches.
6. The tournament progresses until a champion is determined.




Running large-scale Monte Carlo simulations to estimate tournament win probabilities
