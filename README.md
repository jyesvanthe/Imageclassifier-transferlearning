#**IMAGE CLASSIFIER** 

**Overview**<br>
This repository contains an animal image classification model built using FastAI and trained on a dataset of 90 different animal species. The model uses ResNet34 as the backbone and applies progressive resizing, data augmentation, and early stopping to improve performance.

**Dataset**<br>
The dataset is sourced from Kaggle and consists of images of 90 different animals. It is automatically downloaded and cached using KaggleHub.

**Dataset Path:**
```
/root/.cache/kagglehub/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals/versions/5/animals/animals
```

**Features**

FastAI Data Augmentation with optimized transformations.

Progressive Resizing to improve generalization.

ResNet34 CNN architecture for feature extraction.

Early Stopping & Model Saving for efficient training.

Mixed Precision Training (fp16) for faster computation.

Model Export & Inference for easy deployment.

**Installation**

Ensure you have Python 3.7+ and the required dependencies installed.
```
pip install fastai kagglehub
```

 **Training the Model**
```
 python train.py
```

**Training Pipeline:**

Load dataset and apply data augmentations.

Define the ResNet34 model.

Find the best learning rate.

Train using the one-cycle policy with early stopping.

Save the best model automatically.

**Exporting the Model**

After training, the model is exported as animal_classifier_resnet34.pkl for inference.

```
learn.export('animal_classifier_resnet34.pkl')
```
**Making Predictions**

To predict on a new image:

```
from fastai.vision.all import *
img = PILImage.create('test_image.jpg')
learn = load_learner('animal_classifier_resnet34.pkl')
pred_class, pred_idx, probs = learn.predict(img)
print(f'Predicted class: {pred_class}, Probability: {probs[pred_idx]:.4f}')
```
**Model Interpretation**

To analyze model performance and visualize incorrect predictions:

```
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(10,10))

```
**Troubleshooting**

If dataset is not found, ensure you have access to Kaggle datasets.

If training is slow, reduce batch size (bs=32).

For out-of-memory errors, train using GPU with fp16 precision.

**License**

This project is licensed under the MIT License.

**Drive link**<br>
https://drive.google.com/file/d/1LSQ9HQcrnZtJVA0XM47sF2s1T3E5_CMK/view?usp=drive_link

