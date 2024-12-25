# Object Detection Model Using ResNet50

This repository contains an object detection model implemented using the ResNet50 architecture and trained on the [CIFAR-10 dataset](https://www.kaggle.com/c/cifar-10/).

## Overview
The model leverages the ResNet50 deep learning architecture to perform object detection on the CIFAR-10 dataset. CIFAR-10 is a labeled subset of the 80 million tiny images dataset and consists of 60,000 32x32 color images in 10 different classes.

## Features
- **Architecture**: ResNet50 pre-trained on ImageNet, fine-tuned for CIFAR-10.
- **Dataset**: CIFAR-10, including 50,000 training images and 10,000 test images.
- **Task**: Multi-class object detection with bounding box regression.

## Requirements
To run the code and train the model, the following dependencies are required:

- Python 3.8+
- TensorFlow 2.0+ or PyTorch
- NumPy
- OpenCV
- Matplotlib
- Pandas
- Scikit-learn

Install the required libraries using:
```bash
pip install -r requirements.txt
```

## Dataset
1. Download the CIFAR-10 dataset from [Kaggle](https://www.kaggle.com/c/cifar-10/) and place it in the `data/` directory.
2. The dataset should include:
   - Training images
   - Test images
   - Class labels

## Model Training
1. Preprocess the dataset:
   - Resize images to fit the ResNet50 input dimensions (224x224).
   - Normalize pixel values to the range [0, 1].
2. Load the ResNet50 model pre-trained on ImageNet.
3. Add custom layers for bounding box regression and classification.
4. Compile the model with appropriate loss functions for detection and classification.
5. Train the model on the CIFAR-10 dataset.

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten

# Load ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = Flatten()(base_model.output)
classification_output = Dense(10, activation='softmax', name='classification')(x)
bbox_output = Dense(4, activation='linear', name='bounding_box')(x)

# Final model
model = Model(inputs=base_model.input, outputs=[classification_output, bbox_output])
model.compile(
    optimizer='adam',
    loss={
        'classification': 'categorical_crossentropy',
        'bounding_box': 'mean_squared_error'
    },
    metrics={
        'classification': 'accuracy',
        'bounding_box': 'mse'
    }
)
```

## Evaluation
Evaluate the model on the CIFAR-10 test set to calculate:
- **Classification Accuracy**
- **Mean Squared Error (MSE)** for bounding box predictions.

## Results
The model achieved the following results:
- Classification accuracy: *0.9380*

## Usage
To use the trained model for inference:
1. Load the saved model weights.
2. Provide input images for prediction.
3. Display predicted bounding boxes and class labels.

```python
from tensorflow.keras.models import load_model
import cv2

# Load model
model = load_model('resnet50_cifar10.h5')

# Predict
image = cv2.imread('test_image.jpg')
image_resized = cv2.resize(image, (224, 224))
predictions = model.predict(image_resized)
```

## References
- [CIFAR-10 Dataset on Kaggle](https://www.kaggle.com/c/cifar-10/)
- [ResNet50 Paper](https://arxiv.org/abs/1512.03385)
