# 🏥 Enterprise Medical Image Diagnosis & DICOM Clinical Platform

An end-to-end clinical decision support platform powered by Deep Learning (ResNet50), Explainable AI (Grad-CAM), DICOM processing, real-time risk triage, and automated report generation.

---

## 🌟 Key Features

* **Multi-Format Ingestion:** Supports DICOM (`.dcm`), JPG, and PNG radiographs.
* **Deep Neural Inference:** Calibrated disease classification with confidence index.
* **Explainable AI (Grad-CAM):** Visual heatmap overlay highlighting diagnostic focus regions.
* **Interactive Image Tools:** Real-time brightness & contrast adjustment sliders.
* **Clinical Triage & Severity:** Automated classification (LOW/NORMAL, MODERATE, CRITICAL).
* **Database History Tracking:** Persistent patient case storage using SQLite3.
* **Dual-Scan Progression:** Side-by-side comparative analysis of historical scans.
* **Automated PDF Export:** 1-Click generation of formal hospital report documents with physician notes and signatures.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit, Plotly
* **AI & Computer Vision:** PyTorch, torchvision, OpenCV
* **Medical Data:** Pydicom, Pillow
* **Database & Export:** SQLite3, ReportLab
