# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:31:09 2020

@author: Admin
"""
import streamlit as st
from PIL import Image
from plwo import predict1

st.title("CLASSIFY YOUR FOOD IMAGE")
st.write("lets predict your food")
uploaded_file=st.file_uploader("Choose an image--",type="jpg")
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,caption="uploaded image",use_column_width=True)
    st.write("predicting your food-- it takes a while")
    st.write("")
    label=predict1(uploaded_file)
    st.write("your food is.............(loading)")
    st.write(label)
    st.write('here it is!')