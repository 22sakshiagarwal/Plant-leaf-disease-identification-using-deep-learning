# Plant Leaf Disease Identification

This project utilizes a Convolutional Neural Network (CNN) with the VGG19 architecture to identify diseases in plant leaves.

## Dataset

The dataset used for training, validation, and testing the model is quite large and, therefore, stored on Google Drive.

**You can download the dataset from the following link:**

[Plant Leaf Disease Dataset](https://drive.google.com/drive/folders/1aOw7VeR6FisNwNKV5RWQXhHWw7PINO2R?usp=drive_link)

**Instructions for accessing the dataset:**

1.  Click on the link provided above.
2.  You will be redirected to a Google Drive folder containing the dataset.
3.  Download the entire folder or the necessary subfolders (train, validation, test).
4.  Ensure that the folder structure within the downloaded dataset is maintained, as the code expects a specific directory organization.  The expected structure is:

    ```
    dataset/
    ├── train/
    │   ├── Class_1_Disease/
    │   │   └── [image files]
    │   ├── Class_2_Disease/
    │   │   └── [image files]
    │   └── ...
    ├── validation/
    │   ├── Class_1_Disease/
    │   │   └── [image files]
    │   ├── Class_2_Disease/
    │   │   └── [image files]
    │   └── ...
    └── test/
        ├── Class_1_Disease/
        │   └── [image files]
        ├── Class_2_Disease/
        │   └── [image files]
        └── ...
    ```
    (Replace `Class_1_Disease`, `Class_2_Disease`, etc., with the actual names of the plant and disease categories in your dataset.)

5.  Place the downloaded `dataset` folder in the appropriate directory relative to where you run the notebook (e.g.,  if your notebook expects the data in `/content/drive/MyDrive/data/`,  make sure the `dataset` folder ends up there).  You may need to adjust the paths in the notebook to match your local setup.

## Project Overview

The `plant_leaf_disease_identification.ipynb` notebook contains the code for building and training the CNN model.  Here's a summary of the process:

* **Data Loading and Preprocessing:** The notebook uses `ImageDataGenerator` from Keras to efficiently load and augment the images.  This includes resizing, normalization, and data augmentation techniques (zoom, shear, flips) to improve model robustness.
* **Model Architecture:** The VGG19 model, pre-trained on ImageNet, is used as the base.  The convolutional base is frozen to leverage learned feature extractors, and a new classification head (Flatten and Dense layers) is added and trained to classify the plant diseases.
* **Training:** The model is trained using the Adam optimizer and categorical cross-entropy loss.  Early stopping and model checkpoint callbacks are used to prevent overfitting and save the best-performing model.
* **Evaluation:** The trained model can be used to predict the disease of a given plant leaf image.

## Dependencies

The project relies on the following Python libraries:

* tensorflow/keras
* pandas
* numpy
* matplotlib
* os

You can install these libraries using pip:

```bash
pip install tensorflow pandas numpy matplotlib
