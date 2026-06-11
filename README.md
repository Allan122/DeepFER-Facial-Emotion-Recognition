# DeepFER: Facial Emotion Recognition Using Deep Learning 🧠📸

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![Keras](https://img.shields.io/badge/Keras-3.0-red.svg)
![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-success.svg)

## Project Overview
The **DeepFER (Facial Emotion Recognition)** project is an advanced Computer Vision pipeline designed to accurately classify real-time human facial expressions into seven distinct emotional states: **Angry, Sad, Happy, Fear, Neutral, Disgust, and Surprise**. 

By leveraging Convolutional Neural Networks (CNNs), Data Augmentation, and Cost-Sensitive Learning, this system overcomes the limitations of traditional handcrafted feature extraction. It provides a robust, empathetic machine-learning tool capable of processing real-time visual data for applications in customer service, mental health monitoring, and human-computer interaction.

## Problem Statement
Automated systems currently struggle to accurately understand and respond to human emotions. This limits the effectiveness of virtual assistants, automated customer service routing, and telehealth platforms. Relying on post-interaction surveys is highly reactive and suffers from massive response bias. 

**The Objective:** Design, train, and deploy an advanced deep learning architecture capable of predicting user emotions in real-time directly from raw webcam/image pixels. This shifts the operational framework from a *reactive* cost center to a *proactive*, emotionally aware asset.

## Dataset Overview
The project utilized a massive dataset of over **71,000 grayscale facial images** (48x48 pixels).
* **Target Classes (7):** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
* **Data Wrangling:** Engineered a pipeline using `cv2` to systematically scan and filter out corrupted files and zero-variance outliers (pitch-black/blank images) prior to model ingestion.
* **Class Imbalance Handling:** The dataset was heavily biased toward 'Happy' and 'Neutral'. Instead of using SMOTE (which blurs image interpolations), **Cost-Sensitive Learning (Class Weights)** was mathematically calculated and injected into the training loop to penalize the loss function for majority-class bias.

## Dataset Link
https://drive.google.com/file/d/1zpTI5zE0R_jREra7jw6THfZ-RXIeN7sm/view?usp=drive_link

## Tech Stack & Libraries
* **Core Language:** Python
* **Deep Learning Framework:** TensorFlow & Keras
* **Computer Vision:** OpenCV (`cv2`)
* **Hyperparameter Tuning:** KerasTuner (Bayesian Optimization, Hyperband)
* **Data Manipulation & Math:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Joblib (Label Encoder Serialization), Keras `.h5` (Model Weights)

## Key Methodologies

### 1. Dynamic Data Augmentation
To prevent the CNN from overfitting to perfectly centered faces, an `ImageDataGenerator` was deployed. This synthesized infinite real-world variations by applying random rotations (15°), width/height shifts, zooming, and horizontal flipping dynamically during the training batches. Raw pixel intensities were normalized from `[0, 255]` to `[0, 1]` to ensure rapid gradient descent.

### 2. Deep Learning Architecture Engineering
Three distinct CNN topologies were engineered and rigorously tested:
* **Model 1 (Baseline CNN):** Standard convolutional blocks with basic Dropout. Tuned via RandomSearch.
* **Model 2 (Deep CNN + Batch Norm):** An advanced, deep architecture utilizing `BatchNormalization` after every Conv2D layer to prevent internal covariate shift. Tuned via **Bayesian Optimization** to find the perfect dense layer width.
* **Model 3 (LeakyReLU CNN):** Replaced standard ReLUs with `LeakyReLU` to prevent the "Dying ReLU" gradient problem. Tuned via Hyperband.

### 3. Model Explainability (Grad-CAM)
To break the "black box" stigma of deep learning, **Gradient-weighted Class Activation Mapping (Grad-CAM)** was engineered. The resulting heatmaps mathematically proved that the CNN was actively firing its neurons on the structural geometry of the face (the eyes, mouth, and brow ridges) to make its predictions, rather than memorizing background noise.

## Results & Evaluation
* **Winning Architecture:** Model 2 (Bayesian-Tuned Deep CNN with Batch Normalization).
* **Primary Metrics:** Evaluated primarily on **Macro F1-Score** and **Recall** to ensure minority classes (like Disgust and Fear) were not ignored. 
* **Performance:** Achieved highly stable training and validation loss curves, successfully extracting complex non-linear spatial features while aggressively neutralizing overfitting through heavy dropout (0.5) and dynamic augmentation.

## Business Impact & Deployment
The fully optimized model and corresponding categorical encoder were serialized into `.h5` and `.joblib` formats. 
* **Customer Service:** Real-time routing of frustrated or angry users to specialized human retention agents.
* **Telehealth:** Instant flagging of subtle depressive or fearful micro-expressions for clinical review.
* **UX/UI:** Enables software to adapt dynamically to the user's current emotional state, fostering deeper human-computer empathy.

## Author
**Allan Cheerakunnil Alex** Aspiring Data Scientist | AI & Deep Learning Enthusiast  
*Master's Program Candidate at AlmaBetter*

---
