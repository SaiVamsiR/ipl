import streamlit as st
import pickle
import pandas as pd

# Load the pre-trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title('🏏 IPL Win Predictor! 🏆')

teams = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Gujarat Titans',
    'Kolkata Knight Riders',
    'Lucknow Super Giants',
    'Mumbai Indians',
    'Punjab Kings',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]

city = sorted([
    'Abu Dhabi', 'Ahmedabad', 'Bangalore', 'Bengaluru', 'Bloemfontein', 'Cape Town', 'Centurion', 'Chandigarh',
    'Chennai', 'Cuttack', 'Delhi', 'Dharamsala', 'Dubai', 'Durban', 'East London', 'Hyderabad', 'Indore',
    'Jaipur', 'Johannesburg', 'Kanpur', 'Kimberley', 'Kochi', 'Kolkata', 'Mohali', 'Mumbai', 'Nagpur',
    'Navi Mumbai', 'Port Elizabeth', 'Pune', 'Raipur', 'Rajkot', 'Ranchi', 'Sharjah', 'Visakhapatnam'
])

st.write("---")

# Team selection with 🆚 in center
st.header("Teams Selection")
col1, col_center, col2 = st.columns([5, 1, 5])
with col1:
    batting_team = st.selectbox('🏏 Batting Team', sorted(teams))
with col_center:
    st.markdown("<h3 style='text-align: center;'>🆚</h3>", unsafe_allow_html=True)
with col2:
    bowling_team = st.selectbox('🎳 Bowling Team', sorted(teams))

if batting_team == bowling_team:
    st.warning("Oops! Batting and Bowling teams cannot be the same. Please select different teams. ⚠️")

st.write("---")

# Match info
st.header("Match Details 🏟️")
selected_city = st.selectbox('🏙️ Host City', city)
target = st.number_input('🎯 Target Score (1st Innings)', min_value=0, step=1)

st.write("---")

# Match progress
st.header("Current Match Progress 📊")
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('📈 Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('⏱️ Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('📉 Wickets Fallen', min_value=0, max_value=10, step=1)

st.write("---")

# Prediction
if st.button('Predict Win Probability ✨'):
    if overs > 20:
        st.error("Overs cannot exceed 20. ❌")
    elif batting_team == bowling_team:
        st.error("Select different teams for batting and bowling before predicting. ⚠️")
    else:
        runs_left = target - score
        balls_left = max(0, 120 - int(overs * 6))
        wickets_remaining = 10 - wickets
        crr = (score / overs) if overs > 0 else 0
        rrr = (runs_left * 6 / balls_left) if balls_left > 0 else (runs_left * 6)

        input_df = pd.DataFrame({
            'BattingTeam': [batting_team],
            'BowlingTeam': [bowling_team],
            'City': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_remaining],
            'total_run_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        st.subheader("Prediction Results")

        if win > loss:
            st.markdown(f"<h4 style='color:green;'>🏏 {batting_team} Win Probability: {round(win * 100)}%</h4>", unsafe_allow_html=True)
            st.progress(int(win * 100))

            st.markdown(f"<h4 style='color:red;'>🎳 {bowling_team} Win Probability: {round(loss * 100)}%</h4>", unsafe_allow_html=True)
            st.progress(int(loss * 100))
        else:
            st.markdown(f"<h4 style='color:green;'>🎳 {bowling_team} Win Probability: {round(loss * 100)}%</h4>", unsafe_allow_html=True)
            st.progress(int(loss * 100))

            st.markdown(f"<h4 style='color:red;'>🏏 {batting_team} Win Probability: {round(win * 100)}%</h4>", unsafe_allow_html=True)
            st.progress(int(win * 100))

        st.success("May the best team win! 🥳🎉")

st.write("---")
st.info("💡 This predictor uses a machine learning model trained on historical IPL data. Results are probabilistic and not guaranteed.")
