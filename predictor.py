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

# Team selection
st.header("Teams Selection")
col1, col_center, col2 = st.columns([5, 1, 5])
with col1:
    batting_team = st.selectbox('🏏 Batting Team', sorted(teams))
with col_center:
    st.markdown("<h3 style='text-align:center;'>🆚</h3>", unsafe_allow_html=True)
with col2:
    bowling_team = st.selectbox('🎳 Bowling Team', sorted(teams))

if batting_team == bowling_team:
    st.warning("Oops! Batting and Bowling teams cannot be the same. Please select different teams. ⚠️")

st.write("---")

# Match info
st.header("Match Details 🏟️")
selected_city = st.selectbox('🏙️ Host City', city)
target = st.number_input('🎯 Target Score (1st Innings)', min_value=1, step=1)
max_overs = st.number_input('🔢 Max Overs in Match', min_value=1, max_value=20, step=1)

st.write("---")

# Match progress input
st.header("Current Match Progress 📊")
col3, col4 = st.columns(2)
with col3:
    completed_overs = st.number_input('⏱️ Overs Completed (full overs)', min_value=0, max_value=max_overs, step=1)
with col4:
    balls_in_current_over = st.number_input('⚪ Balls in Current Over (0–5)', min_value=0, max_value=5, step=1)

total_balls_bowled = int(completed_overs * 6 + balls_in_current_over)
max_balls = max_overs * 6
balls_left = max(0, max_balls - total_balls_bowled)
overs_float = completed_overs + balls_in_current_over / 6

col5, col6 = st.columns(2)
with col5:
    score = st.number_input('📈 Current Score', min_value=0, step=1)
with col6:
    wickets = st.number_input('📉 Wickets Fallen', min_value=0, max_value=10, step=1)

st.write("---")

# Prediction button
if st.button('Predict Win Probability ✨'):
    if batting_team == bowling_team:
        st.error("Teams must be different for batting and bowling.")
    elif total_balls_bowled > max_balls:
        st.error(f"Overs exceed the max limit of {max_overs}. Adjust inputs.")
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

        st.success("May the best team win! 🎉")

st.write("---")
st.info("💡 This predictor uses a machine learning model trained on historical IPL data. Results are probabilistic and not guaranteed.")
