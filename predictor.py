import streamlit as st
import pickle
import pandas as pd

# Load the pre-trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Set the title of the application with an emoji
st.title('🏏 IPL Win Predictor! 🏆')

# Define the lists for teams and cities
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

# Note: 'Kings XI Punjab' is a historical name; updated to 'Punjab Kings' for consistency
# Also removed duplicate 'Bangalore' entry and sorted the list

city = [
    'Abu Dhabi', 'Ahmedabad', 'Bangalore', 'Bengaluru', 'Bloemfontein', 'Cape Town', 'Centurion', 'Chandigarh',
    'Chennai', 'Cuttack', 'Delhi', 'Dharamsala', 'Dubai', 'Durban', 'East London', 'Hyderabad', 'Indore',
    'Jaipur', 'Johannesburg', 'Kanpur', 'Kimberley', 'Kochi', 'Kolkata', 'Mohali', 'Mumbai', 'Nagpur',
    'Navi Mumbai', 'Port Elizabeth', 'Pune', 'Raipur', 'Rajkot', 'Ranchi', 'Sharjah', 'Visakhapatnam'
]
# Added 'Kochi' and 'Mohali' as they are also IPL venues, and sorted the list.

st.write("---") # Horizontal line for better visual separation

# Input sections for batting and bowling teams
st.header("Teams Selection 🆚")
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the **Batting Team** 🏏', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the **Bowling Team** 🎳', sorted(teams))

# Ensure the selected teams are different
if batting_team == bowling_team:
    st.warning("Oops! Batting and Bowling teams cannot be the same. Please select different teams. ⚠️")

st.write("---")

# Input section for host city and target score
st.header("Match Details 🏟️🎯")
selected_city = st.selectbox('Select the **Host City** 🏙️', sorted(city))
target = st.number_input('**Target Runs** (set by the first innings) 🎯', min_value=0, step=1)

st.write("---")

# Input section for current match status
st.header("Current Match Progress 📊")
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('**Current Score** (runs scored by batting team) 📈', min_value=0, step=1)
with col4:
    overs = st.number_input('**Overs Completed** ⏱️', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('**Wickets Fallen** 📉', min_value=0, max_value=10, step=1)

st.write("---")

# Prediction button
if st.button('Predict Win Probability! ✨'):
    # Basic validation for overs
    if overs > 20:
        st.error("Overs cannot exceed 20. Please enter a valid value. ❌")
    elif batting_team == bowling_team:
        st.error("Please select different teams for batting and bowling before predicting. 🏏🆚🎳")
    else:
        # Calculate key metrics for prediction
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        
        # Ensure balls_left doesn't go negative if overs exceed 20 or if calculations result in it
        balls_left = max(0, balls_left) 
        
        wickets_remaining = 10 - wickets
        
        # Avoid division by zero for CRR and RRR
        crr = (score / overs) if overs > 0 else 0
        rrr = (runs_left * 6 / balls_left) if balls_left > 0 else (runs_left * 6) # If no balls left, RRR is effectively infinite if runs needed
        
        # Create DataFrame for prediction
        input_df = pd.DataFrame({
            'BattingTeam': [batting_team],
            'BowlingTeam': [bowling_team],
            'City': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_remaining], # Changed to wickets_remaining for clarity as per model expectation
            'total_run_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Make prediction
        result = pipe.predict_proba(input_df)

        # Extract probabilities
        loss = result[0][0]
        win = result[0][1]

        # Display results with emojis and bold text
        st.subheader("Prediction Results:")
        st.metric(label=f"**{batting_team}** Win Probability 🥳", value=f"{round(win * 100)}%")
        st.metric(label=f"**{bowling_team}** Win Probability 💪", value=f"{round(loss * 100)}%")
        
        st.success("May the best team win! 🥳🎉")

st.write("---")
st.info("💡 This predictor uses a machine learning model trained on historical IPL data. Results are probabilistic and not guaranteed.")
