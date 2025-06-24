import streamlit as st
import pickle
import pandas as pd

# Load the pre-trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title('🏏 IPL Winner Predictor! 🏆')

teams = sorted([
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 'Kolkata Knight Riders',
    'Lucknow Super Giants', 'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
])

city = sorted([
    'Abu Dhabi', 'Ahmedabad', 'Bangalore', 'Bengaluru', 'Bloemfontein', 'Cape Town', 'Centurion', 'Chandigarh',
    'Chennai', 'Cuttack', 'Delhi', 'Dharamsala', 'Dubai', 'Durban', 'East London', 'Hyderabad', 'Indore',
    'Jaipur', 'Johannesburg', 'Kanpur', 'Kimberley', 'Kochi', 'Kolkata', 'Mohali', 'Mumbai', 'Nagpur',
    'Navi Mumbai', 'Port Elizabeth', 'Pune', 'Raipur', 'Rajkot', 'Ranchi', 'Sharjah', 'Visakhapatnam'
])

st.write("---")
st.header("Teams Selection")
col1, col_center, col2 = st.columns([5, 1, 5])
with col1:
    batting_team = st.selectbox('🏏 Batting Team', teams)
with col_center:
    st.markdown("<h3 style='text-align:center;'>🆚</h3>", unsafe_allow_html=True)
with col2:
    bowling_team = st.selectbox('🎳 Bowling Team', teams)

if batting_team == bowling_team:
    st.warning("Batting and Bowling teams cannot be the same!")

st.write("---")
st.header("Match Details 🏟️")
selected_city = st.selectbox('🏙️ Host City', city)
target = st.number_input('🎯 Target Runs', min_value=1, step=1)
max_overs = st.number_input('🔢 Maximum Overs in Match', min_value=1, max_value=20, step=1)

st.write("---")
st.header("Current Match Progress 📊")
col3, col4 = st.columns(2)
with col3:
    completed_overs = st.number_input('⏱️ Completed Overs (full)', min_value=0, max_value=max_overs, step=1)
with col4:
    balls_this_over = st.number_input('⚪ Balls in Current Over (0-5)', min_value=0, max_value=5, step=1)

col5, col6 = st.columns(2)
with col5:
    score = st.number_input('📈 Current Score', min_value=0, step=1)
with col6:
    wickets = st.number_input('📉 Wickets Fallen', min_value=0, max_value=10, step=1)

# Calculations
total_balls_bowled = int(completed_overs * 6 + balls_this_over)
max_balls = max_overs * 6
balls_left = max(0, max_balls - total_balls_bowled)
overs_float = completed_overs + balls_this_over / 6

st.write("---")
if st.button('Predict Win Probability ✨'):
    if total_balls_bowled > max_balls:
        st.error("Total overs exceed maximum match overs. Please adjust your input.")
    elif batting_team == bowling_team:
        st.error("Teams must be different!")
    else:
        runs_left = target - score
        wickets_remaining = 10 - wickets
        crr = (score / overs_float) if overs_float > 0 else 0
        rrr = (runs_left * 6 / balls_left) if balls_left > 0 else runs_left * 6

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
        loss, win = result[0][0], result[0][1]

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

        st.success("May the best team win! 🏆")

st.write("---")
st.info("💡 This predictor uses a machine learning model trained on historical IPL data. Results are probabilistic and not guaranteed.")
