# Pl8Catch: The License Plate Tracker

Pl8Catch is a comprehensive license plate recognition system designed to detect vehicles, extract license plate information, and provide a user-friendly interface for tracking and managing license plates.

## Features

- **YOLO Integration**: Utilizes YOLOv9 for efficient and accurate vehicle and license plate detection.
- **OCR (Optical Character Recognition)**: Employs OCR techniques to extract text from license plates.
- **FastAPI Backend**: Wraps the functionality into API using FastAPI for seamless integration into other applications.
- **Streamlit Frontend**: Provides a frontend interface built with Streamlit for easy access and interaction.

## Execution

### Backend

```bash
python src/pl8catch/app.py
```

### Frontend

```bash
streamlit run src/frontend/app.py
```
