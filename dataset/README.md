# Medical Image Diagnosis Platform

A lightweight deep learning web application to analyze chest X-ray images, detect abnormalities, and generate explainable heatmap visualizations for clinical reference.

## Features
- **X-Ray Image Analysis:** Classifies chest X-rays using a ResNet50 CNN model.
- **Grad-CAM Explainability:** Displays a visual heatmap over the X-ray image to highlight regions the model focused on.
- **SQLite Database Integration:** Automatically saves patient details, diagnosis results, confidence scores, and timestamps for case history tracking.
- **PDF Report Generation:** Exports printable clinical diagnostic summaries for patients.
- **Error Handling:** Handles invalid image uploads and missing input fields smoothly.

## Tech Stack
- **Language:** Python
- **Libraries:** Streamlit, PyTorch, Torchvision, OpenCV, ReportLab, Pillow, NumPy
- **Database:** SQLite3

## How to Run

1. Clone or download this project folder.
2. Install the required Python packages:
   ```bash
   pip install streamlit torch torchvision opencv-python pillow reportlab numpy