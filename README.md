HandSafeNet: Real-Time Hand Detection & Safety Alert System

HandSafeNet is an end-to-end Computer Vision and Deep Learning project that detects a user's hand and evaluates its proximity to a safety zone. Based on the distance, the system classifies the status as:

SAFE

WARNING

DANGER

This project includes dataset collection, CNN model training, and a real-time proximity-based alert system.

hand_poc/
│
├── dataset/
│     ├── hand/               # Hand images
│     ├── no-hand/            # Non-hand images
│
├── model/
│     └── hand_model.h5       # Saved trained CNN model
│
├── src/
│     └── capture_dataset.py  # Script for dataset collection
│
├── hand_training.py          # CNN model training script
├── main.py                   # Real-time detection + safety alert logic
│
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation

System Components
1️- Dataset Collection – src/capture_dataset.py

This script lets you collect images for training the classifier.
Controls:

H → Save hand image

N → Save no-hand image

Q → Quit

Images are automatically stored in:

dataset/hand/
dataset/no-hand/

2️- Model Training – hand_training.py

This script trains a Convolutional Neural Network (CNN) to classify images as:

Hand (1)

No-Hand (0)

Features used during training:

Image augmentation

Batch normalization

Dropout

Early stopping

Model checkpointing

The final trained model is saved at:

model/hand_model.h5

3️- Real-Time Detection – main.py

This script performs:

✔ Skin color auto-calibration
✔ HSV mask generation
✔ Contour & centroid detection
✔ Distance measurement
✔ Safety-zone classification
✔ Live visualization
Status Logic
Status	Condition	Color
SAFE	Distance ≥ 350px	Green
WARNING	150px–349px	Yellow
DANGER	<150px OR hand enters box	Red
Installation
1️- (Optional) Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Mac/Linux

2️- Install dependencies
pip install -r requirements.txt

How to Run the Project
Step 1 — Capture Dataset
python src/capture_dataset.py

Step 2 — Train the Model
python hand_training.py

Step 3 — Run Real-Time Detection
python main.py

Technologies Used
Purpose	Library
Image Processing	OpenCV
Neural Networks	TensorFlow / Keras
Numerical Computing	NumPy
Data Handling	Scikit-learn
Visualization	OpenCV overlays
Use Cases

Industrial machinery safety

Driver monitoring systems

Human–machine interaction

Touchless gesture interfaces

Hazard distance monitoring

Author

Shivanshu Gangwar
Machine Learning & Computer Vision Developer

📌 Professional Summary

HandSafeNet demonstrates a full ML workflow: dataset creation → CNN training → real-time computer vision system with logical decision-making.
