🚦 Traffic Sign Classification Using Images

A Deep Learning based Traffic Sign Classification system developed using Python, TensorFlow/Keras CNN, CustomTkinter and MySQL.

📌 Project Description

This project identifies and classifies traffic signs from images using a Convolutional Neural Network (CNN). The user can select a traffic sign image through the graphical user interface, and the system predicts the corresponding traffic sign class and displays its name along with the prediction confidence.

✨ Features

* Traffic sign image selection
* CNN-based image classification
* 58 traffic sign classes
* Sign name prediction using label mapping
* Prediction confidence percentage
* Attractive CustomTkinter GUI
* MySQL database integration
* Prediction history storage
* Model evaluation
* Confusion matrix
* Accuracy, precision, recall and F1-score evaluation

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* OpenCV
* Pillow
* CustomTkinter
* MySQL
* Scikit-learn
* Matplotlib
🧠 Machine Learning Model

The project uses a Convolutional Neural Network (CNN) for traffic sign image classification.

Input image size:

`32 × 32 × 3`

The images are normalized before being passed to the CNN model.

📊 Model Performance

Validation Accuracy:

**99.88%**

The project also evaluates the model using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

📁 Project Structure


TrafficSignClassification/
│
├── database/
├── model/
├── prediction/
├── training/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md


▶️ How to Run

1. Install Python

Install Python 3.x.

2. Install required packages


pip install -r requirements.txt

3. Configure MySQL

Create the required MySQL database and update the database connection settings in:


database/db_connect.py


4. Run the application


python main.py


📷 Prediction

The user selects a traffic sign image and the system displays:

Predicted Sign: <Sign Name>
Confidence: <Percentage>
Class ID: <Class ID>

👩‍💻 Project

Traffic Sign Classification Using Images

Developed as an internship project.

