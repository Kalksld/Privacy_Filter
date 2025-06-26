# --- Import libraries ---
import streamlit as st
import re
import cv2
import numpy as np
import pytesseract

# --- Tesseract Path (IMPORTANT) ---
# Change this path if your tesseract is installed elsewhere
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Define sensitive patterns ---
EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
PHONE_REGEX = r"\b\d{10}\b|\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b"
DOOR_REGEX = r"\b\d{1,3}[-/]\d{1,4}(/\d{1,4})?\b|\bH\.?No\.?\s*\d{1,4}\b"
LOCATION_KEYWORDS = ["home", "address", "street", "colony", "road", "apartment", "city", "vizag", "hyderabad"]
BAD_WORDS = ["damn", "hate", "kill", "stupid"]

# --- Function to check privacy ---
def check_privacy(post):
    found = []

    if re.search(EMAIL_REGEX, post):
        found.append("Email detected")

    if re.search(PHONE_REGEX, post):
        found.append("Phone number detected")

    if re.search(DOOR_REGEX, post, re.IGNORECASE):
        found.append("Door number / House number detected")

    for word in LOCATION_KEYWORDS:
        if word.lower() in post.lower():
            found.append(f"Location word detected: '{word}'")

    for bad in BAD_WORDS:
        if bad.lower() in post.lower():
            found.append(f"Inappropriate word detected: '{bad}'")

    return found

# --- Streamlit Web App Title ---
st.title("Privacy Filter for Social Media Posts")

# --- Section 1: Text Post Check ---
st.header("1. Check Privacy in Text Post")

post = st.text_area("Paste your social media text post here:")

if st.button("Check Text Privacy"):
    if post:
        issues = check_privacy(post)
        if issues:
            st.warning("Privacy Issues Found in Text:")
            for item in issues:
                st.write(f"- {item}")
        else:
            st.success("✅ Text post is safe to share!")
    else:
        st.info("Please enter some text to check.")

# --- Section 2: Image Post Check ---
st.header("2. Check Privacy in Image Post")

uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR", caption="Original Image", use_column_width=True)

    # Face Detection and Blurring
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
            image[y:y+h, x:x+w] = blurred_face
        st.image(image, channels="BGR", caption="Faces Blurred", use_column_width=True)

    # OCR: Extract text from image
    extracted_text = pytesseract.image_to_string(image)
    st.subheader("Extracted Text from Image:")
    st.code(extracted_text)

    # Privacy Check on Extracted Text
    st.subheader("Privacy Check Result from Image Text:")
    if extracted_text.strip():
        issues_found = check_privacy(extracted_text)
        if issues_found:
            st.warning("Privacy Issues Found in Image:")
            for issue in issues_found:
                st.write(f"- {issue}")
        else:
            st.success("✅ No privacy issues found in the image!")
    else:
        st.info("No readable text found in the uploaded image.")
