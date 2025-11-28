# app.py
import streamlit as st
from inference import infer_frame, infer_video
import cv2
import tempfile
import os

st.set_page_config(page_title="Fall Detection", layout="centered")
st.title("Fall Detection")

mode = st.radio("Choose mode", ["Live Camera (snapshot)", "Upload Video"])

if mode == "Live Camera (snapshot)":
    st.markdown("This captures a snapshot from your webcam and runs detection. Live streaming is limited on Streamlit Cloud; this is a snapshot mode.")
    img_file = st.camera_input("Take a photo (allow camera access)")
    if img_file is not None:
        with st.spinner("Running detection..."):
            bytes_data = img_file.getvalue()
            # convert bytes to numpy image
            import numpy as np
            nparr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            res = infer_frame(frame)
        st.json(res)
        if res.get("fall"):
            st.error(f"Fall detected! Type: {res.get('type')} (confidence: {res.get('confidence')})")
        else:
            st.success("No fall detected.")

else:
    st.markdown("Upload a video file (mp4/avi/mov). We will sample frames to speed up inference.")
    uploaded = st.file_uploader("Upload video", type=["mp4","avi","mov"])
    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.getbuffer())
        tfile.flush()
        with st.spinner("Processing video (this may take a while on CPU)..."):
            summary = infer_video(tfile.name, sample_rate=5)
        st.json(summary)
        if summary.get("fall_detected"):
            st.warning("Fall detected in uploaded video.")
        os.unlink(tfile.name)
