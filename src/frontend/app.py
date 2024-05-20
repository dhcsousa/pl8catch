import streamlit as st
import requests
from PIL import Image
import numpy as np
import io

# Streamlit app title
st.title("Pl8Catch - Vehicle Detection and License Plate Recognition")


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
        bytes_data = bytes()
        # Create a placeholder for the image
        image_placeholder = st.empty()
        for chunk in response.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b"\xff\xd8")
            b = bytes_data.find(b"\xff\xd9")
            if a != -1 and b != -1:
                jpg = bytes_data[a : b + 2]
                bytes_data = bytes_data[b + 2 :]
                frame = Image.open(io.BytesIO(jpg))
                frame_array = np.array(frame)
                # Update the image in the placeholder
                image_placeholder.image(frame_array, caption="Video Stream", use_column_width=True)


# Call the function with the FastAPI video endpoint URL
backend_endpoint = "http://127.0.0.1:8000/video-detection"
fetch_video_frames(backend_endpoint)
