# Samosa and Burger Classification

## Overview
This project is a deep learning-based image classification model that distinguishes between **samosas** and **burgers** using Convolutional Neural Networks (CNNs). The model is built using TensorFlow and Keras and trained on a dataset of labeled images of samosas and burgers.

## Features
- Uses a **CNN model** with multiple Conv2D and MaxPooling layers.
- Classifies images into **two categories: Samosa and Burger**.
- Achieves competitive accuracy through **training and validation**.
- Provides a **user-friendly interface** for image classification.

## Dataset
The dataset consists of images of samosas and burgers, divided into **training**, **validation**, and **testing** sets. Data augmentation techniques can be applied to improve model performance.

## Model Architecture
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
```

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- TensorFlow and Keras
- NumPy, Pandas, Matplotlib

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/DharmikPrajapati23/Samosa-and-Burger-Classification.git
   cd samosa-burger-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Classifying an Image
To classify a new image, run:
```python
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    class_labels = ['Samosa', 'Burger']
    return class_labels[np.argmax(prediction)], prediction[0]
```

## Deployment
This model can be deployed using **Flask**, **FastAPI**, or **Streamlit**. Example deployment steps will be added soon.

## Contributing
Feel free to fork this repository and submit pull requests for improvements!

## License
This project is licensed under the MIT License.

---
Let me know if you need any updates! 🚀

