import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load the pre-trained model ---
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'pipe.pkl' not found. Please ensure the trained model file is in the correct directory.")
    st.stop()

# --- Hardcoded lists for teams and cities ---
# These lists must accurately reflect the categories the model was trained on
teams = sorted([
    'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad',
    'Delhi Capitals', 'Chennai Super Kings', 'Gujarat Titans',
    'Lucknow Super Giants', 'Kolkata Knight Riders', 'Punjab Kings',
    'Mumbai Indians', 'Rising Pune Supergiants'
])

cities = sorted([
    'Ahmedabad', 'Kolkata', 'Mumbai', 'Navi Mumbai', 'Pune', 'Dubai',
    'Sharjah', 'Abu Dhabi', 'Delhi', 'Chennai', 'Hyderabad',
    'Visakhapatnam', 'Chandigarh', 'Bengaluru', 'Jaipur', 'Indore',
    'Bangalore', 'Kanpur', 'Rajkot', 'Raipur', 'Ranchi', 'Cuttack',
    'Dharamsala', 'Nagpur', 'Johannesburg', 'Centurion', 'Durban',
    'Bloemfontein', 'Port Elizabeth', 'Kimberley', 'East London',
    'Cape Town'
])

# --- Streamlit App Interface ---
st.set_page_config(page_title="IPL Win Predictor")
st.title('IPL Win Predictor !!')

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the Batting Team', teams)

with col2:
    bowling_team = st.selectbox('Select the Bowling Team', teams)

selected_city = st.selectbox('Select Host City', cities)

target = st.number_input('Target Score', min_value=1, value=150)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Current Score', min_value=0, value=0)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=19.5, step=0.1, value=0.0)
with col5:
    wickets_fallen = st.number_input('Wickets Fallen', min_value=0, max_value=10, value=0)


if st.button('Predict Win Probability'):
    # Ensure batting and bowling teams are different
    if batting_team == bowling_team:
        st.error("Batting and Bowling teams cannot be the same. Please select different teams.")
    else:
        runs_left = target - score
        balls_left = 120 - int(overs * 6)

        # 'wickets' in the input DataFrame should represent wickets fallen, consistent with model training
        wickets_for_model = wickets_fallen

        # Current Run Rate (CRR) calculation: (current_score * 6) / balls_played
        balls_played = int(overs * 6)
        if balls_played == 0:
            crr = 0.0 # Avoid division by zero at the start of the innings
        else:
            crr = (score * 6) / balls_played

        # Required Run Rate (RRR) calculation
        if balls_left == 0:
            rrr = 0.0 # Avoid division by zero if all balls are bowled or target already met
        else:
            rrr = (runs_left * 6) / balls_left
            if rrr < 0 : # Handle cases where runs_left is negative (target exceeded)
                rrr = 0.0

        # Create input DataFrame with original categorical columns
        # The 'pipe' object is expected to handle the one-hot encoding internally
        input_df = pd.DataFrame({
            'BattingTeam': [batting_team],
            'BowlingTeam': [bowling_team],
            'City': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_for_model],
            'total_run_x': [target], # total_run_x corresponds to the target score from the notebook
            'crr': [crr],
            'rrr': [rrr]
        })

        # Directly pass input_df to the pipe. The pipe's internal ColumnTransformer
        # is expected to handle the one-hot encoding of 'BattingTeam', 'BowlingTeam', 'City'.
        result = pipe.predict_proba(input_df)

        # Extract probabilities
        loss_prob = round(result[0][0] * 100)
        win_prob = round(result[0][1] * 100)

        st.header(f"{batting_team} - {win_prob}%")
        st.header(f"{bowling_team} - {loss_prob}%")
