import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tempfile
import os
import numpy as np
from keras.utils import load_img, img_to_array
from tempfile import NamedTemporaryFile
from keras.models import load_model
from PIL import Image

model = load_model('Improved_capsule_network.h5')
# st.set_page_config(layout = 'centered')

st.markdown("""
<style>
body {
  background: #ff0099; 
}
</style>
    """, unsafe_allow_html=True)

st.title("Intrusion forger detection using CapsuleNet")
st.subheader("About the Model")
st.write("""Capsule Networks (CapsNets) present a revolutionary paradigm in deep learning architecture, offering a unique solution to challenges
          encountered by traditional convolutional neural networks (CNNs). Introduced
          by Geoffrey Hinton and colleagues in 2017, CapsNets introduce capsules, which
          encode hierarchical relationships and spatial hierarchies within images.
          Capsules enable dynamic routing-by-agreement, allowing capsules in different layers to establish meaningful connections based on spatial coherence. .""")

st.subheader("How to Use")
st.write("1.Upload an image_file")
st.write("2.The image_file should be of a person")
st.write("3.It'll tell if it's a Forged image_file or not")



image_file = st.file_uploader('Upload a image_file file', type = ['jpg','png','jpeg'])



if image_file is None:
    st.text('Upload an image file')

else:

    suffix = image_file.name.split('.')[1]
    # st.write(suffix)

    with NamedTemporaryFile(dir='.',suffix='.'+suffix,delete = False) as f:
        f.write(image_file.getbuffer())
        test_image_file = image.load_img(f.name, target_size = (224, 224))
        test_image_file = np.asarray(test_image_file)
        test_image_file = np.expand_dims(test_image_file, axis=0)
        # st.write(f.name)

    st.image(image_file)
    os.remove(f.name)

    predict = model.predict(test_image_file)

    label = np.argmax(predict)

    st.write(predict[0][label])

    st.header("Fake" if label == 0 else "Real")
    st.subheader("With a probability of {}%".format(round(float(predict[0][label]*100),2)))
    # st.balloons()

    st.header(predict)