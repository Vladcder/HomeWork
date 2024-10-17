import os
import pandas as pd
import streamlit as st
import folium as fl
from streamlit_folium import st_folium
from src.utils import prepare_data, train_model, read_model




st.set_page_config(
    page_title="Apartment prices App",
)
location = [55.742691, 37.586404]
def get_pos(lat1, lng1):
    return lat1, lng1
m = fl.Map(location=[55.742691, 37.586404], min_lat=55.468426, max_lat=56.028824, min_lon=37.136489,
               max_lon=38.122467)
fl.Rectangle([(55.468426, 37.136489), (56.028824, 38.122467)]).add_to(m)
m.add_child(fl.LatLngPopup())
map = st_folium(m, height=350, width=700)
if map.get("last_clicked"):
    location_from_map = get_pos(map["last_clicked"]["lat"], map["last_clicked"]["lng"])
    location = location_from_map




model_path = 'lr_fitted.pkl'
area = st.sidebar.number_input("Specify the area of the apartment", 5	, 2500, 15)
rooms = st.sidebar.number_input("Specify numbers of the rooms", 1	, 20, 2)
floor = st.sidebar.number_input("Specify floor", 1	, 100, 5)
if ((location[0]>56.028824 or location[0]<55.468426) or (location[1]>38.122467 or location[1]<37.136489)):
    lat = st.sidebar.number_input("Specify latitude", 55.468426, 56.028824, 55.742691)
    lon = st.sidebar.number_input("Specify longitude", 37.136489, 38.122467, 37.136489)
    st.sidebar.write("Location on map out of bounds, map center selected")
else:
    lat = st.sidebar.number_input("Specify latitude", 55.468426, 56.028824, location[0])
    lon = st.sidebar.number_input("Specify longitude", 37.136489, 38.122467, location[1])


inputDF = pd.DataFrame(
     {
         "lat":lat,
         "lon":lon,
         "total_square": area,
         "rooms": rooms,
         "floor": floor,
     },
     index=[0],
 )

if not os.path.exists(model_path):
    train_data = prepare_data()
    train_data.to_csv('data.csv')
    train_model(train_data)

if(st.sidebar.button("Predict the price")):
    model = read_model('lr_fitted.pkl')
    preds = model.predict(inputDF)
    st.sidebar.write(f"Model result")
    st.sidebar.write(f"Apartment price is: " + str(round(preds[0])) + " Rub")