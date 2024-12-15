# Pulmonary Disease Classification using ResNet50

## Overview

This project implements a deep learning model to classify chest X-ray images into three categories: Tuberculosis, Pneumonia, and Normal using the ResNet50 architecture. The model is trained on augmented images, fine-tuned using transfer learning, and evaluated for classification accuracy.

## Dataset

The dataset consists of chest X-ray images for three classes:

Tuberculosis
Pneumonia
Normal
The data has been preprocessed and augmented to balance the classes and improve the model's performance.


## Dataset Preparation

Organize the images in a directory structure as follows:
all_images/
    ├── imgs/
        ├── Tuberculosis/
        ├── Pneumonia/
        ├── Normal/

        
## Data Augmentation

The dataset is augmented to increase its size and variability, addressing class imbalance. The following augmentations were applied:

  Rotation (up to 10 degrees)
  Width and height shifts (up to 10%)
  Shear and zoom ranges (up to 10%)
  Horizontal flipping
  Brightness adjustments
  Model Architecture


### This project uses the ResNet50 model pre-trained on ImageNet for transfer learning:

Base model: ResNet50 (pre-trained, not trainable)
Global Average Pooling: Reduces the output dimensions of the convolutional layers.
Fully Connected Layer: Dense layer with 1024 units and ReLU activation.
Output Layer: Softmax activation for multi-class classification (3 classes).
Fine-Tuning
After training the model initially with the base layers frozen, some layers of the ResNet50 model were unfrozen and fine-tuned to improve performance.


## Training the Model

The model is trained with the following configuration:

Optimizer: Adam
Learning rate: 1e-4 (initial), fine-tuned to 1e-5
Loss Function: Categorical Crossentropy
Metrics: Accuracy
Callbacks

### Early Stopping: Stops training if validation loss does not improve for 5 epochs.
### Learning Rate Reduction: Reduces the learning rate by a factor of 0.2 when validation loss plateaus.


## Evaluation

The model's performance is evaluated on a validation set, achieving a validation accuracy of 88.19% and validation loss of 0.6203 after fine-tuning.

