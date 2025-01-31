# Plant Disease Detection and classification using ResNet50

This project utilizes a pre-trained ResNet50 model fine-tuned on a plant disease dataset to classify images of plant leaves as either diseased or healthy.  It leverages data augmentation techniques to improve model robustness and generalization.  The model is trained to identify various plant diseases, offering a potential solution for early detection and intervention in agriculture.

## Dataset

The dataset used for training and validation is the "New Plant Diseases Dataset(Augmented)" available on Kaggle. It contains images of various plant leaves with different diseases, categorized into different classes.  The dataset is split into training, validation, and test sets.  Data augmentation is applied to the training set to increase the diversity of the training data and improve model performance.
## 📂 Dataset
🔗 **Download the dataset:** [New Plant Diseases Dataset(Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)


## Model Architecture

The model is based on the ResNet50 architecture, a deep convolutional neural network known for its strong performance in image classification tasks.  The pre-trained ResNet50 model (trained on ImageNet) is used as a base, with its top classification layers removed.  Custom layers are added on top, including:

*   Global Average Pooling: Reduces the spatial dimensions of the feature maps.
*   Dense Layers: Fully connected layers for further feature processing.
*   Dropout: A regularization technique to prevent overfitting.

This approach allows us to leverage the learned features of ResNet50 while adapting the model to the specific task of plant disease classification.

## Training and Evaluation

The model is trained using the Adam optimizer with a learning rate of 0.001.  Callbacks such as `ReduceLROnPlateau` (for dynamic learning rate adjustment) and `EarlyStopping` (to prevent overfitting) are employed during training.  The model's performance is evaluated on the validation set.

## Prediction

To make predictions on new images, the images are preprocessed (rescaled and normalized) and fed into the trained model. The model outputs class probabilities, and the class with the highest probability is selected as the predicted class. The prediction is then interpreted as either "Diseased" or "Non-Diseased" based on whether the predicted label contains the word "healthy".

## How to Run the Code

1.  **Clone the Repository:**  Clone this GitHub repository to your local machine.
2.  **Download the Dataset:** Download the "New Plant Diseases Dataset(Augmented)" from Kaggle and place it in the appropriate directory (specified in the code).
3.  **Install Dependencies:** Ensure you have the required libraries installed (e.g., TensorFlow, Keras, NumPy, Matplotlib). You can install them using pip:
    ```bash
    pip install tensorflow keras numpy matplotlib
    ```
4.  **Run the Notebook:** Open and run the Jupyter Notebook (`Detection.ipynb`) provided in the repository.  The notebook contains all the code for data loading, model training, and prediction.

## Example Usage

The notebook demonstrates how to load a single test image or loop through all images in a test directory for prediction.  It displays the original image along with the predicted label and the "Diseased"/"Non-Diseased" prediction result.

## Results

The results of the model's training and prediction are displayed in the notebook.  The accuracy and loss curves are plotted to visualize the training progress.  Example predictions on test images are shown.

## Future Work

*   Explore other pre-trained models (e.g., EfficientNet, Inception).
*   Implement Grad-CAM or other visualization techniques to understand which parts of the image are most influential in the model's decision.
*   Deploy the model for real-time prediction (e.g., using a web application or mobile app).

## Acknowledgements

*   Kaggle for providing the dataset.
*   The developers of ResNet50 and other open-source libraries used in this project.
