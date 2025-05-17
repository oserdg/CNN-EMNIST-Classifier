# CNN-EMNIST-Classifier
A deep learning project using a Convolutional Neural Network to classify handwritten letters and digits from the EMNIST dataset.

## Overview
This project implements a Convolutional Neural Network (CNN) for classifying characters from the EMNIST (Extended MNIST) dataset. The EMNIST dataset is an extension of the classic MNIST dataset, containing handwritten digits (0-9) and letters (A-Z, a-z) for a total of 47 balanced classes.

### Dataset Information
The EMNIST dataset used in this project is the "EMNIST Balanced" dataset which contains:
  -112,800 training images
  -18,800 test images
  -Each image is 28x28 pixels grayscale
  -47 balanced classes (digits 0-9 and uppercase letters A-Z)
The dataset files used are:
  -emnist-balanced-train-images-idx3-ubyte
  -emnist-balanced-train-labels-idx1-ubyte
  -emnist-balanced-test-images-idx3-ubyte
  -emnist-balanced-test-labels-idx1-ubyte

## Model Architecture
  The CNN model consists of the following layers:
  
  *1.Input Layer*: Accepts 28x28 grayscale images
  
  *2.Convolutional Layers*:
  
    -Conv2D (32 filters, 3x3 kernel, ReLU activation)
    -MaxPooling2D (2x2 pool size)
    -Conv2D (64 filters, 3x3 kernel, ReLU activation)
    -MaxPooling2D (2x2 pool size)
    -Conv2D (128 filters, 3x3 kernel, ReLU activation)
  
  *3.Flatten Layer*: Converts 3D feature maps to 1D vector
  
  *4.Dense Layers*:
    -Dense (128 units, ReLU activation)
    -Dense (47 units, softmax activation for multi-class classification)
  *Total parameters: 246,319*

## Training Process
  Optimizer: *Adam*
  Loss Function: *Categorical Crossentropy*
  Metrics: Accuracy
  Batch Size: 128
  Epochs: 10
  Training Data: 112,800 images
  Validation Data: 18,800 images

The model achieves:

  -Training accuracy: ~91.25%
  -Validation accuracy: ~87.41%

## Evaluation
The model was evaluated on the test set with the following results:
  Test Loss: *0.378*
  Test Accuracy: *87.41%*

Sample predictions show the model correctly classifying most test images with high confidence, though some confusion occurs between similar-looking characters (e.g., 'O' vs '0', 'I' vs '1').

## Results
The model demonstrates strong performance on the EMNIST dataset, achieving over 87% accuracy on the test set.Most errors occur between visually similar characters.

Example predictions show:
  -Correct classifications with high confidence (>90%)
  -Some misclassifications between similar characters
  -Good generalization from training to test data

## Dependencies
  -Python 3.x
  -TensorFlow 2.x
  -Keras
  -NumPy
  -Matplotlib
  -pandas

### References
    EMNIST Dataset: Cohen et al., 2017
    TensorFlow Documentation: https://www.tensorflow.org/
    Keras Documentation: https://keras.io/
