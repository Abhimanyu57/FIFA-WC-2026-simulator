import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# -------------------------------
# DATA LOADING
# -------------------------------

results = pd.read_csv("results.csv")
shootouts = pd.read_csv("shootouts.csv")
former = pd.read_csv("former_names.csv")
qualified = pd.read_csv("qualified_teams.csv")


# -------------------------------
# DATA CLEANING
# -------------------------------

results["date"] = pd.to_datetime(results["date"])

name_map = dict(zip(former["former"], former["current"]))

results["home_team"] = results["home_team"].replace(name_map)
results["away_team"] = results["away_team"].replace(name_map)

results = results[results["tournament"] != "Friendly"]
results = results[results["date"] >= "2022-12-19"]


# -------------------------------
# CORE TEAMS
# -------------------------------

core_teams = qualified["Team/Slot"].dropna().astype(str).tolist()


# -------------------------------
# QUALIFIER CONTENDER TEAMS
# -------------------------------

qualifier_pool = [
"Curacao","Honduras","Panama","Iraq","Oman",
"Uzbekistan","Gabon","Congo","Bolivia","Peru",
"Uganda","Bahrain","Jordan","Vietnam","Kenya"
]


# -------------------------------
# TARGET VARIABLE
# -------------------------------

def get_result(row):
    if row["home_score"] > row["away_score"]:
        return 2
    elif row["home_score"] < row["away_score"]:
        return 0
    else:
        return 1

results["result"] = results.apply(get_result, axis=1)


# -------------------------------
# TEAM STATS
# -------------------------------

teams = pd.unique(results[['home_team','away_team']].values.ravel())

team_stats = {}

for team in teams:

    home_games = results[results["home_team"] == team]
    away_games = results[results["away_team"] == team]

    scored = home_games["home_score"].sum() + away_games["away_score"].sum()
    conceded = home_games["away_score"].sum() + away_games["home_score"].sum()

    matches = len(home_games) + len(away_games)

    if matches == 0:
        continue

    team_stats[team] = {
        "attack": scored / matches,
        "defense": conceded / matches
    }


# -------------------------------
# FEATURE ENGINEERING
# -------------------------------

features = []
targets = []

for _, row in results.iterrows():

    home = row["home_team"]
    away = row["away_team"]

    if home not in team_stats or away not in team_stats:
        continue

    attack_diff = team_stats[home]["attack"] - team_stats[away]["attack"]
    defense_diff = team_stats[home]["defense"] - team_stats[away]["defense"]

    features.append([attack_diff, defense_diff])
    targets.append(row["result"])

X = np.array(features)
y = np.array(targets)


# -------------------------------
# MODEL TRAINING
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05
)

model.fit(X_train, y_train)


# -------------------------------
# MATCH PREDICTOR
# -------------------------------

def predict_match(teamA, teamB):

    attack_diff = team_stats.get(teamA,{"attack":1})["attack"] - team_stats.get(teamB,{"attack":1})["attack"]
    defense_diff = team_stats.get(teamA,{"defense":1})["defense"] - team_stats.get(teamB,{"defense":1})["defense"]

    X_pred = np.array([[attack_diff, defense_diff]])

    probs = model.predict_proba(X_pred)[0]

    probs = probs / probs.sum()

    return probs


# -------------------------------
# STREAMLIT UI
# -------------------------------

st.title("FIFA World Cup 2026 Predictor")


# -------------------------------
# MATCH PREDICTION
# -------------------------------

st.header("Match Prediction")

teamA = st.selectbox("Team A", core_teams)
teamB = st.selectbox("Team B", core_teams)

if st.button("Predict Match"):

    probs = predict_match(teamA, teamB)

    st.write(teamA + " win probability:", round(probs[2],2))
    st.write("Draw probability:", round(probs[1],2))
    st.write(teamB + " win probability:", round(probs[0],2))


# -------------------------------
# WORLD CUP SIMULATION
# -------------------------------

st.header("Simulate World Cup")

if st.button("Run Simulation"):

    qualifier_winners = random.sample(qualifier_pool, 6)

    st.subheader("Qualifier Winners")

    for team in qualifier_winners:
        st.write(team)

    qualified_teams = core_teams + qualifier_winners

    random.shuffle(qualified_teams)

    groups = [qualified_teams[i:i+4] for i in range(0,48,4)]

    st.subheader("Final Groups")

    group_tables = []

    for i,g in enumerate(groups):

        st.write("Group", chr(65+i), ":", g)

        points = {team:0 for team in g}

        matches = [
            (g[0],g[1]),(g[2],g[3]),
            (g[0],g[2]),(g[1],g[3]),
            (g[0],g[3]),(g[1],g[2])
        ]

        for t1,t2 in matches:

            probs = predict_match(t1,t2)

            result = np.random.choice([0,1,2],p=probs)

            if result == 2:
                points[t1]+=3
            elif result == 0:
                points[t2]+=3
            else:
                points[t1]+=1
                points[t2]+=1

        ranking = sorted(points.items(),key=lambda x:x[1],reverse=True)

        group_tables.append(ranking)


    st.subheader("Group Winners")

    top2 = []
    third = []

    for i,table in enumerate(group_tables):

        winner = table[0][0]
        runner = table[1][0]

        st.write("Group",chr(65+i),"winner:",winner)
        st.write("Group",chr(65+i),"runner-up:",runner)

        top2.extend([winner,runner])
        third.append(table[2])


    third_sorted = sorted(third,key=lambda x:x[1],reverse=True)

    best_third = [x[0] for x in third_sorted[:8]]

    teams = top2 + best_third


    round_names = [
        "Round of 32",
        "Round of 16",
        "Quarterfinals",
        "Semifinals",
        "Final"
    ]

    r = 0

    while len(teams) > 1:

        st.subheader(round_names[r])

        next_round = []

        for i in range(0,len(teams),2):

            t1 = teams[i]
            t2 = teams[i+1]

            probs = predict_match(t1,t2)

            result = np.random.choice([0,1,2],p=probs)

            if result == 2:
                winner = t1
            elif result == 0:
                winner = t2
            else:
                winner = random.choice([t1,t2])

            st.write(t1,"vs",t2,"→",winner)

            next_round.append(winner)

        teams = next_round
        r += 1

    st.success("World Cup Champion: " + teams[0])
