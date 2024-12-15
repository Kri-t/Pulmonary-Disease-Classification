# Pulmonary Disease Classification: Pneumonia vs Tuberculosis

## Introduction

In recent years, medical imaging and deep learning have made significant strides in diagnosing diseases, especially in the field of pulmonary health. This project aims to build an automated classification system using ResNet50, a deep convolutional neural network, to distinguish between Pneumonia and Tuberculosis (TB) in chest X-ray images. This classification system aims to support healthcare professionals in their diagnosis by providing fast, accurate, and reliable predictions.

## The Challenge

Both Pneumonia and Tuberculosis are serious pulmonary diseases that have similar symptoms, such as coughing and difficulty breathing. However, their treatment plans and prognosis are different. Manually diagnosing these diseases from chest X-rays is time-consuming, requires expert knowledge, and can be prone to human error. This is where machine learning-based automated classification systems can provide an edge, enabling faster and more consistent diagnoses.

## The Solution

This project builds a machine learning model using ResNet50, which is pre-trained on the ImageNet dataset and fine-tuned for this specific task of distinguishing between Pneumonia and Tuberculosis. The model is trained on a large dataset of chest X-ray images and achieves high accuracy in classifying these diseases. The solution is intended to assist radiologists and healthcare professionals by providing a second opinion and improving diagnostic speed.

## Key Features
### Deep Learning Model:
The model uses ResNet50, a state-of-the-art convolutional neural network (CNN) architecture that is well-suited for image classification tasks.

### Data Augmentation:
To improve the robustness of the model, data augmentation techniques like rotation, zoom, width/height shift, and horizontal flipping are applied on the training images.

### High Accuracy:
After fine-tuning, the model achieves 88.19% validation accuracy, demonstrating its effectiveness in classifying chest X-ray images.

### Transfer Learning:
ResNet50 is used with pre-trained weights from ImageNet, making it easier to achieve good performance even with relatively smaller datasets.

## Impact
### Improved Diagnostic Efficiency:
The model can quickly analyze chest X-ray images, providing an accurate classification in a fraction of the time it would take a radiologist.

### Supporting Healthcare Professionals:
The model acts as an intelligent assistant to healthcare professionals, offering them a second opinion and increasing confidence in diagnosis, especially in areas with limited access to specialized medical experts.

### Scalability:
This model can be scaled and integrated into healthcare systems for mass screening in hospitals, health centers, or even remote areas, helping to identify patients who need immediate attention.
### Accuracy and Consistency:
The machine learning model removes human bias and inconsistency, ensuring that every chest X-ray is analyzed using the same standards and procedures.

## Technologies Used
ResNet50 (Pre-trained model for image classification)
TensorFlow/Keras (Deep learning framework)
OpenCV (Image preprocessing)
NumPy & Pandas (Data manipulation)
Matplotlib (Visualizations and performance plots)
