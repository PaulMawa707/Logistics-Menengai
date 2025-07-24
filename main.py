import streamlit as st
from app2 import run_wialon_uploader
from test2 import run_tonnage_lookup
import base64

st.set_page_config(page_title="Unified Vehicle Tool", layout="wide")

# Convert image to Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Set background image
def set_background():
    background_image = get_base64_image("pexels-pixabay-236722.jpg")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{background_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Show logo at top-right corner
def show_logo_top_right(image_path, width=120):
    logo_base64 = get_base64_image(image_path)
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div></div> <!-- empty left spacer -->
            <div style="margin-right: 1rem;">
                <img src="data:image/png;base64,{logo_base64}" width="{width}">
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ✅ Apply background and logo
set_background()
show_logo_top_right("CT-Logo.jpg", width=120)  # 

st.markdown("<br>", unsafe_allow_html=True)


# Title (appears below logo)
st.title("🚚 Menengai Logistics Dashboard")

# Sidebar navigation
app_choice = st.sidebar.radio("Select Tool", ["📦 Logistics PDF Order Uploader", "🚛 Vehicle Tonnage Lookup"])

if app_choice == "📦 Logistics PDF Order Uploader":
    run_wialon_uploader()
elif app_choice == "🚛 Vehicle Tonnage Lookup":
    run_tonnage_lookup()
