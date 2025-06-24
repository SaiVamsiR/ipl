import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# Load model
pipe = pickle.load(open('all_models.pkl', 'rb'))

# App title
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏")
st.title("🏏 IPL Win Predictor 🇮🇳")

# Teams and cities
teams = [
    'Rajasthan Royals 🩷', 'Royal Challengers Bangalore ❤️🖤', 'Sunrisers Hyderabad 🧡',
    'Delhi Capitals 💙', 'Chennai Super Kings 💛', 'Gujarat Titans 🔵',
    'Lucknow Super Giants 🔷', 'Kolkata Knight Riders 💜', 'Punjab Kings 🔴',
    'Mumbai Indians 🔵', 'Kings XI Punjab 🔴'
]

cities = sorted([
    'Ahmedabad', 'Kolkata', 'Mumbai', 'Navi Mumbai', 'Pune', 'Dubai', 'Sharjah', 'Abu Dhabi',
    'Delhi', 'Chennai', 'Hyderabad', 'Visakhapatnam', 'Chandigarh', 'Bengaluru', 'Jaipur',
    'Indore', 'Bangalore', 'Kanpur', 'Rajkot', 'Raipur', 'Ranchi', 'Cuttack', 'Dharamsala',
    'Nagpur', 'Johannesburg', 'Centurion', 'Durban', 'Bloemfontein', 'Port Elizabeth',
    'Kimberley', 'East London', 'Cape Town'
])

# Team selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('🏏 Select the Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('🎯 Select the Bowling Team', sorted(teams))

selected_city = st.selectbox('📍 Select Host City', cities)

# Match Situation
target = st.number_input('🎯 Target Score', min_value=1)
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('🏏 Current Score', min_value=0)
with col4:
    overs = st.number_input('⏱ Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets_lost = st.number_input('❌ Wickets Lost', min_value=0, max_value=10)

# Predict button
if st.button('🔮 Predict'):

    if overs == 0:
        st.warning("Overs must be greater than 0.")
    else:
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets = 10 - wickets_lost
        crr = score / overs
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Clean team names (remove emojis for prediction)
        def clean_team(name):
            return name.split(' ')[0] if ' ' in name else name

        input_df = pd.DataFrame({
            'BattingTeam': [clean_team(batting_team)],
            'BowlingTeam': [clean_team(bowling_team)],
            'City': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets],
            'total_run_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Prediction
        result = pipe.predict_proba(input_df)[0]
        loss = result[0]
        win = result[1]

        # Output as a bar chart
        prob_df = pd.DataFrame({
            "Team": [batting_team, bowling_team],
            "Win Probability (%)": [win * 100, loss * 100]
        })

        fig = px.bar(prob_df, x="Team", y="Win Probability (%)",
                     color="Team", text_auto=".2f", height=400)
        st.plotly_chart(fig)


        # Show summary
        st.markdown("### 📊 Prediction Summary:")
        st.success(f"🟢 {batting_team} - **{round(win*100, 2)}%** chance to win!")
        st.error(f"🔴 {bowling_team} - **{round(loss*100, 2)}%** chance to win!")
