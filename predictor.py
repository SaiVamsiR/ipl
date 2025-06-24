import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# --- Load the pre-trained model and data (assuming these files exist) ---
# It's crucial to have 'pipe.pkl', 'Matches_Result.csv', and 'Ball_by_Ball.csv'
# in the same directory as your Streamlit app or adjust paths accordingly.
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'pipe.pkl' not found. Please ensure the trained model file is in the correct directory.")
    st.stop()

try:
    match = pd.read_csv('Matches_Result.csv')
    ball = pd.read_csv('Ball_by_Ball.csv')
except FileNotFoundError:
    st.error("Error: 'Matches_Result.csv' or 'Ball_by_Ball.csv' not found. Please ensure data files are in the correct directory.")
    st.stop()

# --- Data Preprocessing (as done in the IPL notebook) ---
total_score = ball.groupby(['ID', 'innings']).sum()['total_run'].reset_index()
total_score = total_score[total_score['innings'] == 1]
match_df = match.merge(total_score[['ID', 'total_run']], left_on='ID', right_on='ID')
d_df = match_df.merge(ball, on='ID')
d_df = d_df[d_df['innings'] == 2]

# Clean up team names as done in the notebook
teams = [
    'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad',
    'Delhi Capitals', 'Chennai Super Kings', 'Gujarat Titans',
    'Lucknow Super Giants', 'Kolkata Knight Riders', 'Punjab Kings',
    'Mumbai Indians'
] # Note: 'Kings XI Punjab' is likely replaced by 'Punjab Kings' later.
# 'Rising Pune Supergiant', 'Gujarat Lions', 'Rising Pune Supergiants', 'Pune Warriors', 'Deccan Chargers', 'Kochi Tuskers Kerala' are also handled.

# Replace old team names with current ones for consistency
d_df['Team1'] = d_df['Team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
d_df['Team2'] = d_df['Team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
d_df['WinningTeam'] = d_df['WinningTeam'].str.replace('Delhi Daredevils', 'Delhi Capitals')
d_df['BattingTeam'] = d_df['BattingTeam'].str.replace('Delhi Daredevils', 'Delhi Capitals')

d_df['Team1'] = d_df['Team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
d_df['Team2'] = d_df['Team2'].str.replace('Team2', 'Sunrisers Hyderabad') # This line seems like a typo in original; fixed assuming it refers to Team2
d_df['WinningTeam'] = d_df['WinningTeam'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
d_df['BattingTeam'] = d_df['BattingTeam'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')


d_df['Team1'] = d_df['Team1'].str.replace('Gujarat Lions', 'Gujarat Titans')
d_df['Team2'] = d_df['Team2'].str.replace('Gujarat Lions', 'Gujarat Titans')
d_df['WinningTeam'] = d_df['WinningTeam'].str.replace('Gujarat Lions', 'Gujarat Titans')
d_df['BattingTeam'] = d_df['BattingTeam'].str.replace('Gujarat Lions', 'Gujarat Titans')

d_df['Team1'] = d_df['Team1'].str.replace('Kings XI Punjab', 'Punjab Kings')
d_df['Team2'] = d_df['Team2'].str.replace('Kings XI Punjab', 'Punjab Kings')
d_df['WinningTeam'] = d_df['WinningTeam'].str.replace('Kings XI Punjab', 'Punjab Kings')
d_df['BattingTeam'] = d_df['BattingTeam'].str.replace('Kings XI Punjab', 'Punjab Kings')

d_df['Team1'] = d_df['Team1'].str.replace('Pune Warriors', 'Rising Pune Supergiants')
d_df['Team2'] = d_df['Team2'].str.replace('Pune Warriors', 'Rising Pune Supergiants')
d_df['WinningTeam'] = d_df['WinningTeam'].str.replace('Pune Warriors', 'Rising Pune Supergiants')
d_df['BattingTeam'] = d_df['BattingTeam'].str.replace('Pune Warriors', 'Rising Pune Supergiants')

d_df = d_df[d_df['BattingTeam'].isin(teams)]
d_df = d_df[d_df['BowlingTeam'].isin(teams)]

# Calculate current score, runs left, balls left, wickets, crr, rrr
d_df['current_score'] = d_df.groupby('ID')['total_run_y'].cumsum()
d_df['runs_left'] = d_df['total_run_x'] - d_df['current_score']
d_df['balls_left'] = 120 - ((d_df['overs'] * 6) + d_df['ballnumber'])
d_df['player_out'] = d_df['player_out'].fillna("0")
d_df['player_out'] = d_df['player_out'].apply(lambda x: x if x == "0" else "1")
d_df['isWicketDelivery'] = pd.to_numeric(d_df['isWicketDelivery'], errors='coerce')
wickets = d_df.groupby('ID')['isWicketDelivery'].cumsum()
d_df['wickets'] = wickets.values
d_df['crr'] = (d_df['current_score'] * 6) / (120 - d_df['balls_left'])
d_df['rrr'] = (d_df['runs_left'] * 6) / d_df['balls_left']

# Handle infinite RRR values (can occur when balls_left is 0)
d_df['rrr'] = d_df['rrr'].replace([np.inf, -np.inf], 0)

# Drop rows where balls_left is 0 for RRR calculation (or where target is already met)
d_df = d_df[d_df['balls_left'] != 0]

# Prepare feature columns for the model
delivery_df = d_df[['BattingTeam', 'BowlingTeam', 'City', 'runs_left', 'balls_left', 'wickets', 'total_run_x', 'crr', 'rrr']]


# --- Streamlit App Interface ---
st.set_page_config(page_title="IPL Win Predictor")
st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the Bowling Team', sorted(teams))

cities = sorted(d_df['City'].unique())
selected_city = st.selectbox('Select Host City', cities)

target = st.number_input('Target Score', min_value=1)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Current Score', min_value=0)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=19.5, step=0.1)
with col5:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10)

if st.button('Predict Win Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    
    # Ensure balls_left is not zero to avoid division by zero
    if balls_left == 0:
        st.warning("All balls have been bowled. Prediction not possible.")
    else:
        wickets_remaining = 10 - wickets
        
        crr = (score * 6) / (overs * 6 + (overs * 6) % 1) if (overs * 6 + (overs * 6) % 1) > 0 else 0
        
        rrr = (runs_left * 6) / balls_left

        input_df = pd.DataFrame({
            'BattingTeam': [batting_team],
            'BowlingTeam': [bowling_team],
            'City': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_remaining],
            'total_run_x': [target], # This corresponds to the target score
            'crr': [crr],
            'rrr': [rrr]
        })

        # One-hot encode the categorical features
        input_df_encoded = pd.get_dummies(input_df, columns=['BattingTeam', 'BowlingTeam', 'City'], drop_first=True)

        # Align columns with the training data (delivery_df used for column reference)
        # This is crucial because get_dummies might create different columns based on input values
        # Create a reference dataframe with all possible dummy columns from the training data
        all_columns = pd.get_dummies(delivery_df, columns=['BattingTeam', 'BowlingTeam', 'City'], drop_first=True).columns
        
        # Add missing columns to input_df_encoded and fill with 0
        missing_cols = set(all_columns) - set(input_df_encoded.columns)
        for c in missing_cols:
            input_df_encoded[c] = 0

        # Ensure the order of columns is the same as the training data
        input_df_encoded = input_df_encoded[all_columns]
        
        # Make prediction
        result = pipe.predict_proba(input_df_encoded)
        
        loss = round(result[0][0] * 100)
        win = round(result[0][1] * 100)

        st.header(f"{batting_team} - {win}%")
        st.header(f"{bowling_team} - {loss}%")
