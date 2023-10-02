import streamlit as st
from PIL import Image
from fastai import *

st.title ("Is this tomato ripe?")

st.write("This is a deep learning classificiation app to determine if a tomato is ripe or not.")

ripeTomato = Image.open("ripeTomato.jpg")
ripeTomato = ripeTomato.resize((300, 200))

unripeTomato = Image.open("unripeTomato.jpg")
unripeTomato = unripeTomato.resize((300, 200))

# some space

st.write(" ")

st.write("Here are some examples of ripe and unripe tomatoes:")

# show and align images horizontally
col1, col2 = st.columns(2)

with col1:
   st.write("Ripe tomatos")
   st.image(image=ripeTomato, caption="This is a ripe tomato", width=300)

with col2:
   st.write("Unripe tomatos")
   st.image(image= unripeTomato, caption="This is an unripe tomato", width=300)

st.write(" ")
st.write(" ")
st.write("Upload an image of a tomato to determine if it is ripe or not.")

inputImage = st.file_uploader('', type='jpg', key=6)
if inputImage is not None:
    tomatoImage = Image.open(inputImage)
    # tomatoImage = image.resize((300, 200))
    st.image(image=tomatoImage, caption="This is the tomato you uploaded", use_column_width=True)


# get the model and predict
learn = load_learner('ripe_or_not.pkl')

if st.button('Predict'):
    if inputImage is not None:
        is_ripe,_,probs = learn.predict(tomatoImage)
        st.write(f"This is a: {is_ripe}.")
        st.write(f"Probability it's a ripe tomato: {probs[0]:.4f}")
    else:
        st.write("Please upload an image first.")