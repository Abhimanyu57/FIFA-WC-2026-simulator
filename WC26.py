import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


st.title("FIFA World Cup 2026 Predictor")


# -----------------------------
# DATA LOADING
# -----------------------------

results = pd.read_csv("results.csv")
shootouts = pd.read_csv("shootouts.csv")
former = pd.read_csv("former_names.csv")
qualified = pd.read_csv("qualified_teams.csv")


# -----------------------------
# DATA CLEANING
# -----------------------------

results["date"] = pd.to_datetime(results["date"])

name_map = dict(zip(former["former"], former["current"]))

results["home_team"] = results["home_team"].replace(name_map)
results["away_team"] = results["away_team"].replace(name_map)

results = results[results["tournament"] != "Friendly"]
results = results[results["date"] >= "2018-01-01"]


# -----------------------------
# CORE TEAMS
# -----------------------------

core_teams = qualified["Team/Slot"].dropna().astype(str).tolist()


# -----------------------------
# QUALIFIER CONTENDERS
# -----------------------------

qualifier_pool = [
"Curacao","Honduras","Panama","Iraq","Oman",
"Uzbekistan","Gabon","Congo","Bolivia","Peru",
"Uganda","Bahrain","Jordan","Vietnam","Kenya"
]


# -----------------------------
# RESULT TARGET
# -----------------------------

def get_result(row):
    if row["home_score"] > row["away_score"]:
        return 2
    elif row["home_score"] < row["away_score"]:
        return 0
    else:
        return 1

results["result"] = results.apply(get_result, axis=1)


# -----------------------------
# TEAM STATS
# -----------------------------

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


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

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


# -----------------------------
# MODEL TRAINING
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05
)

model.fit(X_train, y_train)


# -----------------------------
# MATCH PREDICTOR
# -----------------------------

def predict_match(teamA, teamB):

    attack_diff = team_stats.get(teamA,{"attack":1})["attack"] - team_stats.get(teamB,{"attack":1})["attack"]
    defense_diff = team_stats.get(teamA,{"defense":1})["defense"] - team_stats.get(teamB,{"defense":1})["defense"]

    X_pred = np.array([[attack_diff, defense_diff]])

    probs = model.predict_proba(X_pred)[0]

    probs = probs / probs.sum()

    return probs


# -----------------------------
# BRACKET UI FUNCTIONS
# -----------------------------

def match_box(team1, team2):

    st.markdown(
        f"""
        <div style="
        border:1px solid #888;
        padding:8px;
        margin:6px;
        border-radius:6px;
        text-align:center;
        ">
        <b>{team1}</b><br>
        vs<br>
        <b>{team2}</b>
        </div>
        """,
        unsafe_allow_html=True
    )


def champion_box(team):

    st.markdown(
        f"""
        <div style="
        border:2px solid gold;
        padding:15px;
        margin:10px;
        border-radius:10px;
        text-align:center;
        font-size:22px;
        font-weight:bold;
        ">
        🏆 Champion<br>{team}
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# MATCH PREDICTION UI
# -----------------------------

st.header("Match Prediction")

teamA = st.selectbox("Team A", core_teams)
teamB = st.selectbox("Team B", core_teams)

if st.button("Predict Match"):

    probs = predict_match(teamA, teamB)

    st.write(teamA + " win probability:", round(probs[2],2))
    st.write("Draw probability:", round(probs[1],2))
    st.write(teamB + " win probability:", round(probs[0],2))


# -----------------------------
# WORLD CUP SIMULATION
# -----------------------------

st.header("Simulate World Cup")

if st.button("Run Simulation"):

    qualifier_winners = random.sample(qualifier_pool, 6)

    qualified_teams = core_teams + qualifier_winners

    random.shuffle(qualified_teams)


    # Round of 32
    r32 = [(qualified_teams[i], qualified_teams[i+1]) for i in range(0,32,2)]


    winners_r32 = []

    for t1,t2 in r32:

        probs = predict_match(t1,t2)
        result = np.random.choice([0,1,2],p=probs)

        if result == 2:
            winners_r32.append(t1)
        elif result == 0:
            winners_r32.append(t2)
        else:
            winners_r32.append(random.choice([t1,t2]))


    # Round of 16
    r16 = [(winners_r32[i], winners_r32[i+1]) for i in range(0,16,2)]

    winners_r16 = []

    for t1,t2 in r16:

        probs = predict_match(t1,t2)
        result = np.random.choice([0,1,2],p=probs)

        if result == 2:
            winners_r16.append(t1)
        elif result == 0:
            winners_r16.append(t2)
        else:
            winners_r16.append(random.choice([t1,t2]))


    # Quarterfinals
    qf = [(winners_r16[i], winners_r16[i+1]) for i in range(0,8,2)]

    winners_qf = []

    for t1,t2 in qf:

        probs = predict_match(t1,t2)
        result = np.random.choice([0,1,2],p=probs)

        if result == 2:
            winners_qf.append(t1)
        elif result == 0:
            winners_qf.append(t2)
        else:
            winners_qf.append(random.choice([t1,t2]))


    # Semifinals
    sf = [(winners_qf[i], winners_qf[i+1]) for i in range(0,4,2)]

    winners_sf = []

    for t1,t2 in sf:

        probs = predict_match(t1,t2)
        result = np.random.choice([0,1,2],p=probs)

        if result == 2:
            winners_sf.append(t1)
        elif result == 0:
            winners_sf.append(t2)
        else:
            winners_sf.append(random.choice([t1,t2]))


    # Final
    final = (winners_sf[0], winners_sf[1])

    probs = predict_match(final[0], final[1])
    result = np.random.choice([0,1,2],p=probs)

    if result == 2:
        champion = final[0]
    elif result == 0:
        champion = final[1]
    else:
        champion = random.choice(final)


    # -------------------------
    # BRACKET DISPLAY
    # -------------------------

    st.subheader("Tournament Bracket")

    col1,col2,col3,col4,col5 = st.columns(5)


    with col1:
        st.write("Round of 32")
        for m in r32:
            match_box(m[0],m[1])


    with col2:
        st.write("Round of 16")
        for m in r16:
            match_box(m[0],m[1])


    with col3:
        st.write("Quarterfinals")
        for m in qf:
            match_box(m[0],m[1])


    with col4:
        st.write("Semifinals")
        for m in sf:
            match_box(m[0],m[1])


    with col5:
        st.write("Final")
        match_box(final[0],final[1])
        champion_box(champion)
