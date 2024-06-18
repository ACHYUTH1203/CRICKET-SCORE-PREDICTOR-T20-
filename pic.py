import streamlit as st
import pickle
import pandas as pd
import xgboost
from xgboost import XGBRegressor
import numpy as np

# Load the pre-trained model pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa',
         'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka']

cities = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town',
          'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban',
          'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion',
          'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton',
          'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi',
          'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts',
          'Cardiff', 'Christchurch', 'Trinidad']

# Set the background image using CSS
page_bg_img = '''
<style>
body {
background-image: url("https://img.freepik.com/premium-photo/night-cricket-stadium-illustration-football-night-stadium-background_94574-17008.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

st.title('Cricket Score Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

city = st.selectbox('Select city', sorted(cities))

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score')
with col4:
    overs = st.number_input('Overs done (works for overs > 5)')
with col5:
    wickets = st.number_input('Wickets out')

last_five = st.number_input('Runs scored in last 5 overs')

if st.button('Predict Score'):
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = current_score / overs

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'current_score': [current_score],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'crr': [crr],
        'last_five': [last_five]
    })

    result = pipe.predict(input_df)
    st.header("Predicted Score - " + str(int(result[0])))
