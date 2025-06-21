import streamlit as st
import os
import tempfile
from detection.yolo_detector import ElephantDetector

# Initialize detector
detector = ElephantDetector()

# UI Setup
st.title("🐘 Asian Elephant Individual Identifier")
uploaded_files = st.file_uploader(
    "Upload images (.jpg, .png, .nrw)",
    type=["jpg", "jpeg", "png", ".nrw"],
    accept_multiple_files=True
)

if uploaded_files:
    results_container = st.container()

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            temp_path = tmpfile.name

            try:
                # Process image
                results = detector.identify(temp_path)

                # Display results
                st.subheader(f"Results for {uploaded_file.name}")
                for result in results:
                    st.markdown(f"- **{result['id']}** (Similarity: {result['score']:.2f})")

            finally:
                os.unlink(temp_path)
