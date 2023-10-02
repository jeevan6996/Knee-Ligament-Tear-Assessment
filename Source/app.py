#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 23:21:43 2023

@author: jeevan
"""


import pickle

import numpy as np
import streamlit as st
import utils

best_mrnet_model_name = 'MRNet_Model3'
best_mrnet_model_cutoff_threshold = 0.429860  # Determined during evaluation on the test set
mrnet_model = utils.load_model_from_disk(best_mrnet_model_name)

best_kneemri_model_name = 'kneeMRI_Model6'
kneemri_model = utils.load_model_from_disk(best_kneemri_model_name)

st.set_page_config(
    page_title="Medical Image Models",
)

st.title("KNEE LIGAMENT ASSESSMENT")

st.subheader("Welcome to Deep Learning based assessment of Knee MRI")

st.text('''
    Jeevan Pawar
    K2242210
    MSc Data Science
    ''')

mri_file = st.file_uploader("Upload MRI",
                            type=['npy', 'pck'],
                            key="mri_file")

mrnet_label = {0: 'Healthy', 1: 'ACL Tear'}
kneemri_label = {0: 'Healthy', 1: 'Partial ACL Tear', 2: 'Complete ACL Tear'}

if mri_file is not None:
    if mri_file.name.endswith('.npy'):
        mri_vol = np.load(mri_file)
    elif mri_file.name.endswith('.pck'):
        mri_vol = pickle.load(mri_file)

    mri_vol = mri_vol.astype(np.float64)  # Change the dtype to float64
    # mri_vol.shape

    predict_button = st.button('Predict')

    if predict_button:
        with st.spinner('Preprocessing...'):
            preprocessed_mri_vol = utils.preprocess_mri(mri_vol)
            # preprocessed_mri_vol.shape

        with st.spinner('Predicting...'):
            mri_vol = np.expand_dims(mri_vol, axis=3)  # Adding extra axis for making it compatible for 3D Convolutions
            # mri_vol.shape
            mrnet_pred_prob = mrnet_model.predict(np.array([preprocessed_mri_vol]))
            print(mrnet_pred_prob)
            mrnet_pred_label = (mrnet_pred_prob[0] >= best_mrnet_model_cutoff_threshold).astype('int')
            print(mrnet_pred_label)

            kneemri_pred_prob = kneemri_model.predict(np.array([preprocessed_mri_vol]))
            print(kneemri_pred_prob)
            kneemri_pred_label = kneemri_pred_prob[0].argmax(axis=-1)
            print(kneemri_pred_label)

            if mrnet_pred_label == 1 and kneemri_pred_label == 0:
                if mrnet_pred_prob[0] > kneemri_pred_prob[0][kneemri_pred_label]:
                    st.write(f'ACL Tear Prediction : **{mrnet_label[mrnet_pred_label[0]]}**')
                    st.warning("Possibility of an ACL tear, unsure about the grade of tear.")
            elif mrnet_pred_label == 0 and kneemri_pred_label > 0:
                if mrnet_pred_prob[0] < kneemri_pred_prob[0][kneemri_pred_label]:
                    st.write(f'ACL Tear Grade Prediction : **{kneemri_label[kneemri_pred_label]}**')
                    st.warning("Possibility of an ACL tear.")
            else:
                st.write(f'ACL Tear Prediction : **{mrnet_label[mrnet_pred_label[0]]}**')
                st.write(f'ACL Tear Grade Prediction : **{kneemri_label[kneemri_pred_label]}**')

    slice_number = st.slider('MRI Slice', min_value=1,
                             max_value=30, value=15) - 1

    img = mri_vol[slice_number, :, :]
    normalized_image_data = (img - img.min()) / (img.max() - img.min())

    with st.columns(3)[1]:
        st.image(normalized_image_data, width=300)

    st.error('Disclaimer : The model predictions are just for reference. Please consult your doctor for treatment.')
