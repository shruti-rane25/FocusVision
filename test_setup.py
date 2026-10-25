import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import flask
import joblib


print("================================")
print(" FocusVision Setup Test")
print("================================")
print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)

print("Pandas version:", pd.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Flask version:", flask.version if hasattr(flask, "version") else "Installed")
print("MediaPipe: Imported successfully")
print("Joblib: Imported successfully")
print("\nAll library imports completed!")
print("FocusVision environment is ready.")

