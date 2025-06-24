import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings

# Suppress InconsistentVersionWarning from scikit-learn if versions mismatch
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- Load the pre-trained model ---
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'pipe.pkl' not found. Please ensure the trained model file is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model (pipe.pkl): {e}")
    st.stop()

# --- Hardcoded lists for teams and cities ---
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
st.title('IPL Win Predictor !! 🏏')

# Input fields for match details
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the Batting Team', teams)

with col2:
    bowling_team = st.selectbox('Select the Bowling Team', teams)

selected_city = st.selectbox('Select Host City', cities)

target = st.number_input('Target Score', min_value=1, value=160,
                         help="The total score set by the first innings team.")

# Separate inputs for Overs and Balls for clarity
col3, col4 = st.columns(2)
with col3:
    full_overs = st.number_input('Full Overs Completed', min_value=0, max_value=19, value=0,
                                 help="Number of complete overs bowled (0-19).")
with col4:
    balls_in_current_over = st.number_input('Balls in Current Over', min_value=0, max_value=5, value=0,
                                            help="Number of balls bowled in the current over (0-5).")

wickets_fallen = st.number_input('Wickets Fallen', min_value=0, max_value=10, value=0,
                                 help="Number of wickets lost by the batting team.")

score = st.number_input('Current Score', min_value=0, value=0,
                            help="Current runs scored by the batting team.")


# Predict button
if st.button('Predict Win Probability'):
    # Input validation
    if batting_team == bowling_team:
        st.error("Batting and Bowling teams cannot be the same. Please select different teams.")
    else:
        # --- Handle match already won scenario ---
        if score >= target:
            st.markdown(f"## **<span style='color:green;'>{batting_team}</span> Win Probability: 100%**", unsafe_allow_html=True)
            st.markdown(f"## **<span style='color:red;'>{bowling_team}</span> Win Probability: 0%**", unsafe_allow_html=True)
        else:
            # Calculate total balls bowled
            balls_bowled = full_overs * 6 + balls_in_current_over
            
            # Calculate derived features for the model
            runs_left = target - score
            balls_left = 120 - balls_bowled

            # 'wickets' feature in the model refers to wickets fallen
            wickets_for_model = wickets_fallen

            # Current Run Rate (CRR) calculation
            if balls_bowled == 0:
                crr = 0.0 # Avoid division by zero at the very start
            else:
                crr = (score * 6) / balls_bowled

            # Required Run Rate (RRR) calculation
            if balls_left <= 0: # Handle cases where all balls are bowled or target is met/exceeded
                rrr = 0.0
            else:
                rrr = (runs_left * 6) / balls_left
                if rrr < 0: # If target is already surpassed (safety check)
                    rrr = 0.0

            # Create a Pandas DataFrame for the input, ensuring column names match the original training data
            input_df = pd.DataFrame({
                'BattingTeam': [batting_team],
                'BowlingTeam': [bowling_team],
                'City': [selected_city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets': [wickets_for_model],
                'total_run_x': [target], # This corresponds to the target score
                'crr': [crr],
                'rrr': [rrr]
            })

            try:
                # The 'pipe' object will handle the one-hot encoding internally.
                result = pipe.predict_proba(input_df)

                # Extract probabilities
                loss_prob = round(result[0][0] * 100)
                win_prob = round(result[0][1] * 100)

                st.markdown(f"## **<span style='color:green;'>{batting_team}</span> Win Probability: {win_prob}%**", unsafe_allow_html=True)
                st.markdown(f"## **<span style='color:red;'>{bowling_team}</span> Win Probability: {loss_prob}%**", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.warning("This might be due to an inconsistency between the loaded model and the input data structure, or an issue with the `pipe.pkl` file itself. "
                           "Please ensure your `pipe.pkl` was generated correctly in the `ipl.ipynb` notebook, "
                           "especially that the ColumnTransformer's steps are instantiated objects (e.g., `OneHotEncoder()`) "
                           "and that scikit-learn versions are consistent.")
