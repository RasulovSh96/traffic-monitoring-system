# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:15:22 2026

@author: bahti
"""
import streamlit as st
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="Trafik Analiz", layout="wide")
st.title("📊 Intellektual Trafik Monitoring Tizimi")
st.sidebar.header("⚙️ Boshqaruv")
uploaded_file=st.sidebar.file_uploader("Video yuklang", type=["mp4","avi"])
@st.cache_resource
def load_model():
    return  YOLO("yolov8n.pt")

model=load_model()

if uploaded_file is not None:
    tfile=tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap=cv2.VideoCapture(tfile.name)
    start_button=st.sidebar.button("Ishga tushirish")
    if start_button:
        frame_placeholder=st.empty()
        LINE_Y=350
        processed_ids=set()
        
        csv_filename="Live_traffic_report.csv"
        with open(csv_filename, mode="w", newline="")as f:
            f.write("ID,Vaqt,Soniya\n")
            
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results=model.track(source=frame_rgb, persist=True, tracker="bytetrack.yaml", classes=[2,3,5,7], verbose=False)
            
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.int().tolist()
                ids=results[0].boxes.id.int().tolist()
                
                current_sec=int(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
                min_sec=f"{current_sec//60:02d}:{current_sec%60:02d}"
                
                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2=box
                    cy = int((y1+y2)/2)
                    
                    if (LINE_Y-15<cy<LINE_Y+15) and (track_id not in processed_ids):
                        processed_ids.add(track_id)
                        with open (csv_filename, mode="a", newline="") as f:
                            f.write(f"{track_id},{min_sec},{current_sec}\n")
                            
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame_rgb, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
            cv2.line(frame_rgb, (0, LINE_Y), (frame_rgb.shape[1], LINE_Y), (255, 0, 0), 3)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
        cap.release()
        st.success("🎉 Tugadi!")
        
        if os.path.exists(csv_filename):
            df = pd.DataFrame(processed_data) if 'processed_data' in locals() else pd.read_csv(csv_filename)
            
            if not df.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📋 Yig'ilgan Ma'lumotlar Jadvali")
                    st.dataframe(df, use_container_width=True)
                with col2:
                    st.subheader("📈 Trafik Zichligi Grafigi")
                    oqim = df.groupby('Vaqt').size()
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(oqim.index, oqim.values, marker='o', color='purple', linewidth=2)
                    ax.set_xlabel("Video Vaqti (Daqiqa:Soniya)")
                    ax.set_ylabel("Mashinalar soni")
                    ax.grid(True, linestyle='--', alpha=0.5)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)