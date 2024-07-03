import streamlit as st
import subprocess
import os
import sys

# Set the page configuration to wide mode
st.set_page_config(layout="wide")


# Add custom CSS for the background image
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("https://images.unsplash.com/photo-1640550444366-b94e5752c479?q=80&w=1e.jpg");
            background-size: cover;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }}
        .container {{
            text-align: center;
        }}
        .title {{
            font-size: 2em;
            color: white;
            margin-bottom: 20px;
        }}
        .button {{
            margin: 10px;
            padding: 15px 25px;
            font-size: 16px;
            cursor: pointer;
        }}
        .start-button {{
            background-color: #4CAF50;
            color: white;
            border: none;
        }}
        .stop-button {{
            background-color: #f44336;
            color: white;
            border: none;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()

sys.path.append('C:/Users/ACER/lib/python3.11/site-packages')

st.title('Sign Language Detection Using Python and Machine Learning')

if st.button('Open Camera'):
    try:
        # Run the test.py script
        result = subprocess.run([sys.executable, 'test.py'], capture_output=True, text=True)
        st.write("Script Output:")
        st.text(result.stdout)
        st.write("Script Error (if any):")
        st.text(result.stderr)
    except Exception as e:
        st.error(f"Error: {str(e)}")