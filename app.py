# Libraries
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(layout="wide", page_title="Uber vs Lyft Price Prediction", page_icon="🚗")

# ML Load Models
lr_model = pickle.load(open('linear_regression.pkl','rb'))
pl2_model = pickle.load(open('polynomial_2d_regression.pkl','rb'))
pl3_model = pickle.load(open('polynomial_3d_regression.pkl','rb'))

lgm_model = pickle.load(open('lgm_price_predictor.pkl','rb'))
cat_model = pickle.load(open('cat_price_predictor.pkl','rb'))
xgb_model = pickle.load(open('xgb_price_predictor.pkl','rb'))

# DL Load Models
# fnn_model = pickle.load(open('fnn_model.pkl','rb'))
# cnn_model = pickle.load(open('cnn_model.pkl','rb'))
# rnn_model = pickle.load(open('rnn_model.pkl','rb'))

fnn_model = load_model('fnn_model.h5')
cnn_model = load_model('cnn_model.h5')
rnn_model = load_model('rnn_model.h5')

cover_img = Image.open("uberlyft.png")

# Title
st.markdown("<h1 style='text-align: center;'>Uber vs Lyft Price Prediction</h1>",unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Made By : <b><a href='https://www.linkedin.com/in/prasadmjadhav2/'>Prasad M. Jadhav</a></b>",
    unsafe_allow_html=True
)
st.image(cover_img)
st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;"/>""",unsafe_allow_html=True)

# Model
st.header("ML Choose Model")
ml_choose_model = st.selectbox(
    "Model",
    ["Linear Regressor","Polynomial 2D Reg","Polynomial 3D Reg","LGBM Regressor","CatBoost Regressor","XGB Regressor"],
    label_visibility="collapsed"
)
st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;"/>""",unsafe_allow_html=True)

st.header("DL Choose Model")
dl_choose_model = st.selectbox(
    "Model",
    ["FNN (Feedforward Neural Network)","CNN (Convolutional Neural Network)","RNN (Recurrent Neural Network)"],
    label_visibility="collapsed"
)
st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;"/>""",unsafe_allow_html=True)

# Input
st.header("Input Data")
col1,col2 = st.columns(2)

with col1:
    product_id = st.selectbox(
        "Product ID",
        [
            "55c66225-fbe7-4fd5-9072-eab1ece5e23e",
            "6c84fd89-3f11-4782-9b50-97c468b19529",
            "6d318bcc-22a3-4af6-bddd-b409bfce1546",
            "6f72dfc5-27f1-42e8-84db-ccc7a75f6969",
            "8cf7e821-f0d3-49c6-8eba-e679c0ebcf6a",
            "997acbb5-e102-41e1-b155-9df7de0a73f2",
            "9a0e7b09-b92b-4c41-9779-2ad22b4d779d",
            "lyft",
            "lyft_line",
            "lyft_lux",
            "lyft_luxsuv",
            "lyft_plus",
            "lyft_premier"
        ]
    )
    name = st.selectbox(
        "Name",
        [
            "Black",
            "Black SUV",
            "Lux",
            "Lux Black",
            "Lux Black XL",
            "Lyft",
            "Lyft XL",
            "Shared",
            "Taxi",
            "UberPool",
            "UberX",
            "UberXL",
            "WAV"
        ]
    )
    distance = st.number_input("Distance",value=100)
    source = st.selectbox(
        "Source",
        [
            "Back Bay",
            "Beacon Hill",
            "Boston University",
            "Fenway",
            "Financial District",
            "Haymarket Square",
            "North End",
            "North Station",
            "Northeastern University",
            "South Station",
            "Theatre District",
            "West End"
        ]
    )
    surge_multiplier = st.number_input("Surge Multiplier",value=0.00)

with col2:
    destination = st.selectbox(
        "Destination",
        [
            "Back Bay",
            "Beacon Hill",
            "Boston University",
            "Fenway",
            "Financial District",
            "Haymarket Square",
            "North End",
            "North Station",
            "Northeastern University",
            "South Station",
            "Theatre District",
            "West End"
        ]
    )
    cab_type = st.selectbox("Cab Type",["Lyft", "Uber"])
    long_summary = st.selectbox(
        "Long Summary",
        [
            "Foggy in the morning",
            "Light rain in the morning and overnight",
            "Light rain in the morning",
            "Light rain until evening",
            "Mostly cloudy throughout the day",
            "Overcast throughout the day",
            "Partly cloudy throughout the day",
            "Possible drizzle in the morning",
            "Rain in the morning and afternoon",
            "Rain throughout the day",
            "Rain until morning, starting again in the evening"
        ]
    )
    short_summary = st.selectbox(
        "Short Summary",
        [
            "Clear",
            "Drizzle",
            "Foggy",
            "Light Rain",
            "Mostly Cloudy",
            "Overcast",
            "Partly Cloudy",
            "Possible Drizzle",
            "Rain"
        ]
    )
    icon = st.selectbox(
        "Icon",
        [
            "clear-day",
            "clear-night",
            "cloudy",
            "fog",
            "partly-cloudy-day",
            "partly-cloudy-night",
            "rain"
        ]
    )

st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;"/>""",unsafe_allow_html=True)

# Prediction
st.header("Prediction")

def prediction(model,product_id,name,distance,source,surge_multiplier,destination,cab_type,long_summary,short_summary,icon):
    if product_id == "55c66225-fbe7-4fd5-9072-eab1ece5e23e":
        product_id = 0
    elif product_id == "6c84fd89-3f11-4782-9b50-97c468b19529":
        product_id = 1
    elif product_id == "6d318bcc-22a3-4af6-bddd-b409bfce1546":
        product_id = 2
    elif product_id == "6f72dfc5-27f1-42e8-84db-ccc7a75f6969":
        product_id = 3
    elif product_id == "8cf7e821-f0d3-49c6-8eba-e679c0ebcf6a":
        product_id = 4
    elif product_id == "997acbb5-e102-41e1-b155-9df7de0a73f2":
        product_id = 5
    elif product_id == "9a0e7b09-b92b-4c41-9779-2ad22b4d779d":
        product_id = 6
    elif product_id == "lyft":
        product_id = 7
    elif product_id == "lyft_line":
        product_id = 8
    elif product_id == "lyft_lux":
        product_id = 9
    elif product_id == "lyft_luxsuv":
        product_id = 10
    elif product_id == "lyft_plus":
        product_id = 11
    elif product_id == "lyft_premier":
        product_id = 12

    if name == "Black":
        name = 0
    elif name == "Black SUV":
        name = 1
    elif name == "Lux":
        name = 2
    elif name == "Lux Black":
        name = 3
    elif name == "Lux Black XL":
        name = 4
    elif name == "Lyft":
        name = 5
    elif name == "Lyft XL":
        name = 6
    elif name == "Shared":
        name = 7
    elif name == "Taxi":
        name = 8
    elif name == "UberPool":
        name = 9
    elif name == "UberX":
        name = 10
    elif name == "UberXL":
        name = 11
    elif name == "WAV":
        name = 12

    if source == "Back Bay":
        source = 0
    elif source == "Beacon Hill":
        source = 1
    elif source == "Boston University":
        source = 2
    elif source == "Fenway":
        source = 3
    elif source == "Financial District":
        source = 4
    elif source == "Haymarket Square":
        source = 5
    elif source == "North End":
        source = 6
    elif source == "North Station":
        source = 7
    elif source == "Northeastern University":
        source = 8
    elif source == "South Station":
        source = 9
    elif source == "Theatre District":
        source = 10
    elif source == "West End":
        source = 11

    if destination == "Back Bay":
        destination = 0
    elif destination == "Beacon Hill":
        destination = 1
    elif destination == "Boston University":
        destination = 2
    elif destination == "Fenway":
        destination = 3
    elif destination == "Financial District":
        destination = 4
    elif destination == "Haymarket Square":
        destination = 5
    elif destination == "North End":
        destination = 6
    elif destination == "North Station":
        destination = 7
    elif destination == "Northeastern University":
        destination = 8
    elif destination == "South Station":
        destination = 9
    elif destination == "Theatre District":
        destination = 10
    elif destination == "West End":
        destination = 11

    if cab_type == "Lyft":
        cab_type = 0
    elif cab_type == "Uber":
        cab_type = 1

    if long_summary == "Foggy in the morning":
        long_summary = 0
    elif long_summary == "Light rain in the morning and overnight":
        long_summary = 1
    elif long_summary == "Light rain in the morning":
        long_summary = 2
    elif long_summary == "Light rain until evening":
        long_summary = 3
    elif long_summary == "Mostly cloudy throughout the day":
        long_summary = 4
    elif long_summary == "Overcast throughout the day":
        long_summary = 5
    elif long_summary == "Partly cloudy throughout the day":
        long_summary = 6
    elif long_summary == "Possible drizzle in the morning":
        long_summary = 7
    elif long_summary == "Rain in the morning and afternoon":
        long_summary = 8
    elif long_summary == "Rain throughout the day":
        long_summary = 9
    elif long_summary == "Rain until morning, starting again in the evening":
        long_summary = 10

    if short_summary == "Clear":
        short_summary = 0
    elif short_summary == "Drizzle":
        short_summary = 1
    elif short_summary == "Foggy":
        short_summary = 2
    elif short_summary == "Light Rain":
        short_summary = 3
    elif short_summary == "Mostly Cloudy":
        short_summary = 4
    elif short_summary == "Overcast":
        short_summary = 5
    elif short_summary == "Partly Cloudy":
        short_summary = 6
    elif short_summary == "Possible Drizzle":
        short_summary = 7
    elif short_summary == "Rain":
        short_summary = 8

    if icon == "clear-day":
        icon = 0
    elif icon == "clear-night":
        icon = 1
    elif icon == "cloudy":
        icon = 2
    elif icon == "fog":
        icon = 3
    elif icon == "partly-cloudy-day":
        icon = 4
    elif icon == "partly-cloudy-night":
        icon = 5
    elif icon == "rain":
        icon = 6

    inputs = pd.DataFrame(
        [[product_id,name,distance,source,surge_multiplier,destination,cab_type,long_summary,short_summary,icon]],
        columns=["product_id","name","distance","source","surge_multiplier","destination","cab_type","long_summary","short_summary","icon"]
    )

    if model == 'Linear Regressor':
        predict = lr_model.predict(inputs)[0]
    elif model == 'Polynomial 2D Reg':
        predict = pl2_model.predict(inputs)[0]
    elif model == 'Polynomial 3D Reg':
        predict = pl3_model.predict(inputs)[0]
    
    elif model == 'LGBM Regressor':
        predict = lgm_model.predict(inputs)[0]
    elif model == 'CatBoost Regressor':
        predict = cat_model.predict(inputs)[0]
    elif model == 'XGB Regressor':
        predict = xgb_model.predict(inputs)[0]

    if predict < 0:
        predict = 0

    return predict

if st.button("Click here to predict"):
    result = prediction(ml_choose_model,product_id,name,distance,source,surge_multiplier,destination,cab_type,long_summary,short_summary,icon)
    st.success(f"The predicted price is ${round(result,2)} US")

# Feedback From

import requests
import streamlit as st
from streamlit_lottie import st_lottie

# Replace with your Lottie animation URL or JSON
feedback_animation_url = "https://cdnl.iconscout.com/lottie/premium/thumb/car-on-road-8287943-6621774.json"

def load_lottieurl(url: str):
    import requests
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load the Lottie animation
feedback = load_lottieurl(feedback_animation_url)

# Create a container for the feedback form and animation
with st.container():
    st.header("Send feedback")
    st.caption("Please fill out this form and click 'Send.' I will receive your message via email.")

    contact_form = """
    <form action="https://formsubmit.co/prasadmjadhav6161@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false"><br><br>
        <input type="text" name="name" placeholder="Your name" required><br><br>
        <input type="email" name="email" placeholder="Your email" required><br><br>
        <textarea name="message" placeholder="Your feedback here" required></textarea><br><br>
        <button type="submit">Send</button>
    </form>
    """

    left_column, right_column = st.columns((1, 1))
    with right_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with left_column:
        if feedback:
            st_lottie(feedback, speed=1, reverse=False, loop=True, quality="low", height=None, width=400, key=None)
        else:
            st.error("Failed to load the Lottie animation")




