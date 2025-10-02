"""Frontend Streamlit APP for pl8catch"""

import base64
import io
import json
import os

from dotenv import load_dotenv
import numpy as np
import requests
import streamlit as st
from PIL import Image

# Streamlit app title
st.title("Pl8Catch - Vehicle Detection and License Plate Recognition")

load_dotenv()

BACKEND_ENDPOINT_ENV_VAR = "PL8CATCH_BACKEND_ENDPOINT"


# Function to fetch video frames from FastAPI endpoint
def fetch_video_frames(backend_endpoint: str) -> None:
    """
    Fetch and display video frames from a FastAPI endpoint.

    Parameters
    ----------
    backend_endpoint : str
        The URL of the FastAPI endpoint that provides the video stream.

    Returns
    ----------
    None

    Notes
    ----------
    This function retrieves video frames from the specified FastAPI endpoint and displays them in real-time using Streamlit.
    The video stream is fetched in chunks, and frames are extracted from the byte stream based on JPEG markers.
    Each frame is then displayed in a Streamlit placeholder, updating dynamically to show the video stream.
    """

    response = requests.get(backend_endpoint, stream=True, timeout=10)
    if response.status_code == 200:
        image_placeholder = st.empty()
        metadata_placeholder = st.empty()

        for chunk in response.iter_lines():
            if chunk:
                # Decode the JSON payload
                data = json.loads(chunk.decode("utf-8").replace("data: ", ""))
                frame_base64 = data["frame"]
                detections = data["detections"]

                # Decode the base64-encoded frame
                frame_bytes = base64.b64decode(frame_base64)
                frame = Image.open(io.BytesIO(frame_bytes))
                frame_array = np.array(frame)

                # Update the image and metadata in Streamlit
                image_placeholder.image(frame_array, caption="Video Stream", use_column_width=True)
                metadata_placeholder.json(detections)


# Call the function with the FastAPI video endpoint URL loaded from environment
backend_endpoint = os.getenv(BACKEND_ENDPOINT_ENV_VAR)
if backend_endpoint:
    fetch_video_frames(backend_endpoint)
else:
    st.error(
        "Backend endpoint not configured. Set the environment variable "
        f"{BACKEND_ENDPOINT_ENV_VAR} in a .env file before running the app."
    )
    st.stop()
