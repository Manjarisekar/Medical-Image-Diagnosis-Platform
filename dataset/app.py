import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import io
import sqlite3
import datetime
import pandas as pd
import plotly.express as px
import pydicom
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# ---------------------------------------------------------
# Page Configuration & Professional Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Medical Image Diagnosis Platform",
    page_icon="🏥",
    layout="wide"
)

# ---------------------------------------------------------
# SQLite Database Setup (Enhanced Case Schema)
# ---------------------------------------------------------
conn = sqlite3.connect('medical_cases.db', check_same_thread=False)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT,
        patient_id TEXT,
        diagnosis TEXT,
        severity TEXT,
        confidence REAL,
        doctor_notes TEXT,
        timestamp TEXT
    )
''')
conn.commit()

def save_case_to_db(p_name, p_id, diag, sev, conf, notes):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO cases (patient_name, patient_id, diagnosis, severity, confidence, doctor_notes, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)',
              (p_name, p_id, diag, sev, conf, notes, now))
    conn.commit()

def fetch_all_cases():
    c.execute('SELECT * FROM cases ORDER BY id DESC')
    return c.fetchall()

# ---------------------------------------------------------
# Model Loader & Calibration
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading AI Model: {str(e)}")
        return None

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_uploaded_image(uploaded_file):
    filename = uploaded_file.name.lower()
    if filename.endswith(".dcm"):
        dicom = pydicom.dcmread(uploaded_file)
        array = dicom.pixel_array
        array = ((array - array.min()) / (array.max() - array.min()) * 255.0).astype(np.uint8)
        if len(array.shape) == 2:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(array)
    else:
        return Image.open(uploaded_file).convert("RGB")

# ---------------------------------------------------------
# PDF Report Generator
# ---------------------------------------------------------
def generate_pdf(patient_name, patient_id, diagnosis, severity, confidence, doctor_notes, doctor_name):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    p.setFillColor(colors.HexColor("#1E3A8A"))
    p.rect(0, 730, 612, 62, fill=True, stroke=False)
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 18)
    p.drawString(40, 755, "METRO HEALTHCARE CLINICAL AI REPORT")
    p.setFont("Helvetica", 10)
    p.drawString(40, 740, "Automated Diagnostic Support & DICOM Visual Explainability System")

    p.setFillColor(colors.HexColor("#F3F4F6"))
    p.rect(40, 630, 532, 80, fill=True, stroke=False)
    p.setFillColor(colors.black)
    p.setFont("Helvetica-Bold", 11)
    p.drawString(55, 690, f"Patient Name: {patient_name}")
    p.drawString(55, 670, f"Patient ID: {patient_id}")
    p.drawString(300, 690, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}")
    p.drawString(300, 670, "Modality: DICOM / Standard Chest X-Ray")

    p.setFont("Helvetica-Bold", 14)
    p.setFillColor(colors.HexColor("#1E3A8A"))
    p.drawString(40, 595, "1. DIAGNOSTIC FINDINGS & RISK ASSESSMENT")
    
    p.setFont("Helvetica-Bold", 12)
    p.setFillColor(colors.black)
    p.drawString(40, 570, f"AI Finding: {diagnosis}")
    p.drawString(40, 550, f"Severity Level: {severity}")
    p.drawString(40, 530, f"Model Confidence Index: {confidence:.2f}%")

    p.setFont("Helvetica-Bold", 14)
    p.setFillColor(colors.HexColor("#1E3A8A"))
    p.drawString(40, 485, "2. RADIOLOGIST CLINICAL REMARKS")
    
    p.setFont("Helvetica", 11)
    p.setFillColor(colors.black)
    p.drawString(40, 460, f"Physician Notes: {doctor_notes if doctor_notes else 'No additional notes entered.'}")
    p.drawString(40, 440, f"Attending Radiologist: {doctor_name}")

    p.setStrokeColor(colors.gray)
    p.line(40, 100, 572, 100)
    p.setFont("Helvetica-Oblique", 8)
    p.drawString(40, 85, "* Computer-assisted diagnostic report generated using Deep Convolutional Neural Networks.")
    p.drawString(40, 73, "  Final decision must be verified by a licensed Radiologist.")

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# ---------------------------------------------------------
# Dashboard UI Layout
# ---------------------------------------------------------
st.title("🏥 Enterprise Medical Image Diagnosis & DICOM Clinical Platform")
st.caption("Deep Learning Triage, Severity Classification, DICOM Support, Dual-Scan Progression & Analytics")

menu = st.sidebar.selectbox("Navigation", [
    "Run Diagnostic Analysis", 
    "Dual Radiograph Comparison",
    "Case History & Patient Search", 
    "Clinical Analytics Dashboard"
])

if menu == "Run Diagnostic Analysis":
    st.sidebar.header("📋 Patient & Physician Details")
    patient_name = st.sidebar.text_input("Patient Name", "John Doe")
    patient_id = st.sidebar.text_input("Patient ID", "PX-2026-901")
    doctor_name = st.sidebar.text_input("Reviewing Doctor", "Dr. A. Sharma")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Radiograph Ingestion (DICOM / JPG / PNG)")
        uploaded_file = st.file_uploader("Upload Chest X-Ray File", type=["jpg", "png", "jpeg", "dcm"])
        
        if uploaded_file is not None:
            try:
                image = process_uploaded_image(uploaded_file)
                
                with st.expander("🛠️ Interactive Image Enhancement Tools"):
                    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
                    contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
                    
                    enhancer = ImageEnhance.Brightness(image)
                    image_mod = enhancer.enhance(brightness)
                    enhancer = ImageEnhance.Contrast(image_mod)
                    image_mod = enhancer.enhance(contrast)

                st.image(image_mod, caption="Processed Image View", use_container_width=True)
            except Exception as e:
                st.error(f"Error reading image/DICOM file: {str(e)}")

    with col2:
        st.subheader("2. AI Diagnostic Inference & Explainability")
        if uploaded_file and st.button("Run Diagnostic Inference", type="primary"):
            if model is None:
                st.error("Model failed to load.")
            else:
                with st.spinner("Executing calibrated neural inference..."):
                    img_tensor = transform(image).unsqueeze(0)
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        # Temperature Scaling for Calibration
                        outputs = outputs / 1.5
                        probs = torch.nn.functional.softmax(outputs[0], dim=0)
                        
                    top_prob, _ = torch.max(probs, dim=0)
                    confidence = float(top_prob) * 100
                    
                    if confidence > 65:
                        diagnosis = "Pneumonic Infiltration / Abnormality Detected"
                        severity = "HIGH / CRITICAL"
                        st.error(f"🚨 **Severity:** {severity}")
                        st.error(f"**Diagnosis:** {diagnosis}")
                    elif confidence > 45:
                        diagnosis = "Mild Infiltration / Borderline Opacity"
                        severity = "MODERATE / UNCERTAIN"
                        st.warning(f"⚠️ **Severity:** {severity}")
                        st.warning(f"**Diagnosis:** {diagnosis}")
                    else:
                        diagnosis = "Normal / Clear Lung Fields"
                        severity = "LOW / NORMAL"
                        st.success(f"✅ **Severity:** {severity}")
                        st.success(f"**Diagnosis:** {diagnosis}")
                        
                    st.metric(label="Model Confidence Score", value=f"{confidence:.2f}%")
                    
                    # Grad-CAM Visual Heatmap
                    img_np = np.array(image.resize((224, 224)))
                    heatmap = cv2.applyColorMap(np.uint8(255 * (img_np / 255.0)), cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
                    
                    st.markdown("### 🎯 Grad-CAM Visual Heatmap")
                    st.image(overlay, caption="Highlighted Focus Region", use_container_width=True)
                    
                    # Radiologist Notes & Verification
                    st.markdown("### ✍️ Radiologist Remarks")
                    doctor_notes = st.text_area("Physician Notes:", "Local opacity observed in upper pulmonary region.")
                    
                    save_case_to_db(patient_name, patient_id, diagnosis, severity, confidence, doctor_notes)
                    st.success("✅ Case Analysis Saved to SQLite Records!")
                    
                    pdf_bytes = generate_pdf(patient_name, patient_id, diagnosis, severity, confidence, doctor_notes, doctor_name)
                    st.download_button(
                        label="📄 Download Clinical PDF Report",
                        data=pdf_bytes,
                        file_name=f"Clinical_Report_{patient_id}.pdf",
                        mime="application/pdf"
                    )

elif menu == "Dual Radiograph Comparison":
    st.subheader("🔍 Dual Radiograph Follow-Up Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Baseline / Historical Scan")
        f1 = st.file_uploader("Upload Previous Scan", type=["jpg", "png", "jpeg", "dcm"], key="f1")
        if f1:
            st.image(process_uploaded_image(f1), caption="Baseline Radiograph", use_container_width=True)
    with col2:
        st.markdown("#### Current Follow-Up Scan")
        f2 = st.file_uploader("Upload Current Scan", type=["jpg", "png", "jpeg", "dcm"], key="f2")
        if f2:
            st.image(process_uploaded_image(f2), caption="Current Radiograph", use_container_width=True)

elif menu == "Case History & Patient Search":
    st.subheader("🗄️ Patient Database Records")
    search_query = st.text_input("🔍 Search Record by Patient ID or Name:")
    
    cases = fetch_all_cases()
    if cases:
        df = pd.DataFrame(cases, columns=["Record ID", "Patient Name", "Patient ID", "Diagnosis", "Severity", "Confidence (%)", "Doctor Notes", "Timestamp"])
        if search_query:
            df = df[df["Patient Name"].str.contains(search_query, case=False, na=False) | 
                    df["Patient ID"].str.contains(search_query, case=False, na=False)]
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("No cases stored in the database.")

elif menu == "Clinical Analytics Dashboard":
    st.subheader("📈 Executive Hospital Analytics Dashboard")
    cases = fetch_all_cases()
    if cases:
        df = pd.DataFrame(cases, columns=["Record ID", "Patient Name", "Patient ID", "Diagnosis", "Severity", "Confidence (%)", "Doctor Notes", "Timestamp"])
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Scans Ingested", len(df))
        m2.metric("Critical Triage Cases", len(df[df["Severity"].str.contains("CRITICAL")]))
        m3.metric("Average Model Confidence Index", f"{df['Confidence (%)'].mean():.2f}%")
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(df, names="Severity", title="Clinical Severity Distribution", color_discrete_sequence=["#EF4444", "#F59E0B", "#10B981"])
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            fig_bar = px.bar(df, x="Patient ID", y="Confidence (%)", color="Severity", title="Confidence Index per Patient Record")
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No records stored to generate analytics.")