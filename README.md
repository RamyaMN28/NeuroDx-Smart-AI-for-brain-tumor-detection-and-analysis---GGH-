# NeuroDx-Smart-AI-for-brain-tumor-detection-and-analysis---GGH-
# Brain Tumor Detection Using Deep Learning

## Introduction
This project was developed as part of the **Google Girl Hackathon (GGH)** under the theme **AI in Medicine**. It is designed to assist healthcare professionals by analyzing MRI scans and providing AI-powered diagnostic support for brain tumors.

## Overview
This project utilizes **a deep learning-based approach leveraging Xception**, a state-of-the-art model optimized for medical image classification. The model is trained to classify brain tumors into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

The model is integrated with **Streamlit**, allowing users to upload MRI images and receive AI-generated predictions.

## Table of Contents
- [Installation](#Installation)
- [Dataset](#Dataset)
- [Model Training](#Model-Training)
- [Deployment](#Deployment)
- [Usage](#Usage)
- [Screenshots](#Screenshots)
- [Results](#Results)
- [Acknowledgments](#Acknowledgments)
- [Documentation](#Documentation)

## Installation
Ensure Python is installed on your system. It is recommended to use a virtual environment before installing the dependencies

### Dependencies
The project requires the following dependencies:

```
tensorflow==2.15.0
numpy
pandas
matplotlib
seaborn
glob2
scikit-learn
Pillow
streamlit
```

## Dataset
The dataset consists of MRI images categorized into four tumor types. Data preprocessing and augmentation were applied before training.

Dataset Link: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Additionally, the CSV file used for statistical analysis was compiled from **reliable medical data sources** and includes **edge cases** to ensure comprehensive coverage of different tumor scenarios.

## Model Training
The model is built using TensorFlow and Keras with the following steps:

1. **Preprocessing**: Images are resized to (299, 299) and normalized.
2. **Model Architecture**: Uses Xception as the base model with custom dense layers for classification.
3. **Training**: Implemented categorical cross-entropy loss with Adamax optimizer.
4. **Evaluation**: Achieved **99% accuracy** on validation data.

### Model Training Code
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall

img_shape = (299, 299, 3)
base_model = tf.keras.applications.Xception(include_top=False, weights="imagenet",
                                            input_shape=img_shape, pooling='max')

model = Sequential([
    base_model,
    Flatten(),
    Dropout(rate=0.3),
    Dense(128, activation='relu'),
    Dropout(rate=0.25),
    Dense(4, activation='softmax')
])

model.compile(Adamax(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(), Recall()])

model.summary()
```

## Model and Data Links
- **Trained Model**: [Google Drive Link](https://drive.google.com/drive/folders/1fEO_I4Z4aKoIqqHUiCKNBWCa7y-61WRd?usp=drive_link)
- **Statistical CSV File**: [Google Drive Link](https://drive.google.com/drive/folders/17eZD08afj1CRMYiHtZaH0tZtJPtVHk9k?usp=sharing)
- **Dataset**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Deployment
- The trained model is stored at: `models/brain_tumor_model_99.h5`
- The statistical CSV file is located at: `shap/brain_tumor_dataset.csv`

To run the **Streamlit** application:
```bash
streamlit run app.py
```

## Usage  
1. Upload an **MRI scan of the brain** for analysis.  
2. Enter patient details (name, age, gender, symptoms).  
3. Click **Diagnose** to receive AI-generated predictions, including:  
   - **Predicted Tumor Type**  
   - **AI Confidence Score**  
   - **Tumor Severity Stage**  
   - **Recommended Treatment**  
   - **5-Year Survival Rate**  
4. A visual comparison of **Prediction Confidence vs. Survival Rate** is displayed.  
5. The system provides a **Diagnosis Complete** message and advises consulting a doctor for further evaluation.  

## Screenshots
### Main Interface
![Main Interface](Main_Interface.png)

### Prediction Output
![Prediction Output](Prediction_Output.png)


## Results
- The model achieves **99% accuracy** on the test dataset.
- Provides survival rate and treatment recommendations based on medical datasets.

## Acknowledgments
- **Dataset Source**: Public MRI datasets.
- **Libraries Used**: TensorFlow, Streamlit, Pandas, Seaborn, Matplotlib.

## Author
Developed by [Ramya M N](https://github.com/RamyaMN28)


## Documentation
For more details, refer to the **[Project Documentation](https://docs.google.com/document/d/1g122HOQDj_yPsOMj25YzfUTlFEim8tw8/edit?usp=sharing&ouid=100250789405485360941&rtpof=true&sd=true)**.

## GitHub Repository
Explore the full project on GitHub: [NeuroDx-Smart-AI-for-brain-tumor-detection-and-analysis---GGH-](https://github.com/RamyaMN28/NeuroDx-Smart-AI-for-brain-tumor-detection-and-analysis---GGH-)

