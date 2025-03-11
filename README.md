# Sarcasm Detection using BERT in PyTorch

## Overview
This project implements a sarcasm detection model using a BERT-based neural network in PyTorch. The model is trained on a dataset of news headlines, with the goal of classifying whether a given headline is sarcastic or not.

## Dataset
The dataset used for training and evaluation is the [News Headlines Dataset for Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection). The dataset contains labeled news headlines, where `1` represents a sarcastic headline and `0` represents a non-sarcastic headline.

## Installation
To run the project, you need to install the required dependencies. Use the following commands:
```sh
pip install transformers torch opendatasets scikit-learn matplotlib pandas
```

## Data Preprocessing
- The dataset is loaded using `pandas`.
- Missing and duplicate values are removed.
- The `article_link` column is dropped as it is not needed.
- The dataset is split into training, validation, and testing sets.
- Headlines are tokenized using `AutoTokenizer` from the Hugging Face Transformers library.

## Model Architecture
The model is built using a pre-trained BERT model. The architecture consists of:
- A frozen BERT encoder.
- A dropout layer.
- Two fully connected layers.
- A final sigmoid activation function for binary classification.

## Training Configuration
- **Batch Size:** 32
- **Epochs:** 25
- **Learning Rate:** 1e-14
- **Loss Function:** Binary Cross Entropy Loss
- **Optimizer:** Adam
- **Evaluation Metric:** Accuracy

## Training Process
1. Tokenization of text using `AutoTokenizer`.
2. Conversion of text to PyTorch tensors.
3. Training the model using the dataset.
4. Validation after each epoch.
5. Performance metrics like loss and accuracy are tracked.

## Evaluation
- The trained model is evaluated on a test set.
- The final accuracy and F1-score are displayed.
- Loss and accuracy graphs for training and validation are plotted.

## Results
- The final accuracy score is reported.
- The model is saved and can be loaded for inference.

## Visualization
Loss and accuracy curves are plotted to analyze training and validation performance over epochs using Matplotlib.
