import streamlit as st
import pickle
import pandas as pd

# Load trained pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title('IPL Win Predictor !!')

teams = ['Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad',
         'Delhi Capitals', 'Chennai Super Kings', 'Gujarat Titans', 'Lucknow Super Giants',
         'Kolkata Knight Riders', 'Punjab Kings', 'Mumbai Indians', 'Kings XI Punjab']

city = ['Ahmedabad', 'Kolkata', 'Mumbai', 'Navi Mumbai', 'Pune', 'Dubai',
        'Sharjah', 'Abu Dhabi', 'Delhi', 'Chennai', 'Hyderabad',
        'Visakhapatnam', 'Chandigarh', 'Bengaluru', 'Jaipur', 'Indore',
        'Bangalore', 'Kanpur', 'Rajkot', 'Raipur', 'Ranchi', 'Cuttack',
        'Dharamsala', 'Nagpur', 'Johannesburg', 'Centurion', 'Durban',
        'Bloemfontein', 'Port Elizabeth', 'Kimberley', 'East London',
        'Cape Town']

# Input UI
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the Bowling Team', sorted(teams))

selected_city = st.selectbox('Select Host City', sorted(city))

target = st.number_input('Target', min_value=1.0)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score', min_value=0.0)
with col4:
    overs = st.number_input('Overs', min_value=0.1, max_value=20.0)
with col5:
    wickets_lost = st.number_input('Wickets', min_value=0.0, max_value=10.0)

# Prediction
if st.button('Predict'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets_lost
    crr = score / overs
    rrr = (runs_left * 6) / balls_left if balls_left != 0 else 0

    # Create input df (convert types to expected ones)
    input_df = pd.DataFrame({
        'BattingTeam': [batting_team],
        'BowlingTeam': [bowling_team],
        'City': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'total_run_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Optional: ensure categorical columns are strings
    input_df['BattingTeam'] = input_df['BattingTeam'].astype(str)
    input_df['BowlingTeam'] = input_df['BowlingTeam'].astype(str)
    input_df['City'] = input_df['City'].astype(str)

    st.subheader("Input Data:")
    st.write(input_df)

    try:
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        st.success('Prediction Complete!')
        st.header(f"{batting_team} - {round(win * 100)}%")
        st.header(f"{bowling_team} - {round(loss * 100)}%")
    except Exception as e:
        st.error("Prediction failed. Please check input format and pipeline.")
        st.exception(e)
