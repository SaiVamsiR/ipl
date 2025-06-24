import streamlit as st
import pickle
import pandas as pd

# Load the trained machine learning pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# App Title
st.title('🏏 IPL Win Predictor! 🏆')

# Team and Venue Options
teams = sorted([
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 'Kolkata Knight Riders',
    'Lucknow Super Giants', 'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
])

cities = sorted([
    'Abu Dhabi', 'Ahmedabad', 'Bangalore', 'Bengaluru', 'Bloemfontein', 'Cape Town', 'Centurion',
    'Chandigarh', 'Chennai', 'Cuttack', 'Delhi', 'Dharamsala', 'Dubai', 'Durban', 'East London',
    'Hyderabad', 'Indore', 'Jaipur', 'Johannesburg', 'Kanpur', 'Kimberley', 'Kochi', 'Kolkata',
    'Mohali', 'Mumbai', 'Nagpur', 'Navi Mumbai', 'Port Elizabeth', 'Pune', 'Raipur', 'Rajkot',
    'Ranchi', 'Sharjah', 'Visakhapatnam'
])

# Custom function for colored progress bar
def render_colored_progress(label, percent, color):
    st.markdown(
        f"""
        <div style="margin-bottom:10px">
            <strong>{label}: {percent}%</strong>
            <div style="background-color: #e0e0e0; border-radius: 10px; overflow: hidden;">
                <div style="width: {percent}%; background-color: {color}; padding: 8px 0; text-align: center; color: white;">
                    {percent}%
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("---")
st.header("Team Selection")

col1, col_center, col2 = st.columns([5, 1, 5])
with col1:
    batting_team = st.selectbox("🏏 Batting Team", teams)
with col_center:
    st.markdown("<h3 style='text-align:center;'>🆚</h3>", unsafe_allow_html=True)
with col2:
    bowling_team = st.selectbox("🔴 Bowling Team", teams)

if batting_team == bowling_team:
    st.warning("Teams must be different!")

st.write("---")
st.header("Match Details 🏟️")
venue = st.selectbox("🏙️ Match City", cities)
target = st.number_input("🎯 Target Runs", min_value=1, step=1)
max_overs = st.number_input("🔢 Maximum Overs in Match", min_value=1, max_value=20, value=20)

st.write("---")
st.header("Current Match Situation")

col3, col4 = st.columns(2)
with col3:
    completed_overs = st.number_input("⏱️ Overs Completed (full only)", min_value=0, max_value=max_overs, step=1)
with col4:
    balls_in_current_over = st.number_input("⚪ Balls in Current Over (0–5)", min_value=0, max_value=5, step=1)

col5, col6 = st.columns(2)
with col5:
    score = st.number_input("📈 Current Score", min_value=0, step=1)
with col6:
    wickets = st.number_input("📉 Wickets Fallen", min_value=0, max_value=10, step=1)

# Calculate match metrics
total_balls_bowled = completed_overs * 6 + balls_in_current_over
balls_left = max(0, max_overs * 6 - total_balls_bowled)
overs_float = completed_overs + balls_in_current_over / 6

st.write("---")

if st.button("Predict Win Probability ✨"):
    if total_balls_bowled > max_overs * 6:
        st.error("Overs exceed match limit. Please check your inputs.")
    elif batting_team == bowling_team:
        st.error("Batting and Bowling teams must be different.")
    else:
        runs_left = target - score
        wickets_remaining = 10 - wickets
        crr = score / overs_float if overs_float > 0 else 0
        rrr = (runs_left * 6 / balls_left) if balls_left > 0 else runs_left * 6

        # Prepare DataFrame
        input_df = pd.DataFrame({
            'BattingTeam': [batting_team],
            'BowlingTeam': [bowling_team],
            'City': [venue],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_remaining],
            'total_run_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Predict
        probabilities = pipe.predict_proba(input_df)[0]
        win_percent = round(probabilities[1] * 100)
        loss_percent = round(probabilities[0] * 100)

        st.subheader("Prediction Results 📊")

        if win_percent >= loss_percent:
            render_colored_progress(f"🏏 {batting_team} Win Probability", win_percent, "green")
            render_colored_progress(f"🔴 {bowling_team} Win Probability", loss_percent, "red")
        else:
            render_colored_progress(f"🔴 {bowling_team} Win Probability", loss_percent, "green")
            render_colored_progress(f"🏏 {batting_team} Win Probability", win_percent, "red")

        st.success("May the best team win! 🏆")

st.write("---")
st.info("💡 This model was trained on historical IPL data. Predictions are for fun and educational purposes only.")
