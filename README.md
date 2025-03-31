# Image to Prompt 

## Overview
This project aims to convert images into descriptive text prompts using a combination of deep learning models. The core idea is to map images and their corresponding text prompts into a shared embedding space, allowing us to generate text prompts from images based on their semantic similarity.

## Problem Statement
Given a dataset of images and their corresponding text prompts, the goal is to train a model that can generate a text prompt for a new, unseen image. This involves:
1. Encoding images into embeddings using a pre-trained image encoder (ResNet50).
2. Encoding text prompts into embeddings using a pre-trained text encoder (BERT).
3. Training a model to map image embeddings to text embeddings.
4. Evaluating the model by comparing the similarity between predicted and actual text embeddings.

## Solution
The solution involves the following steps:
1. **Data Preparation**: Load images and their corresponding text prompts from a CSV file.
2. **Model Architecture**:
   - Use a pre-trained ResNet50 model to encode images into embeddings.
   - Use a pre-trained BERT model to encode text prompts into embeddings.
   - Train a neural network to map image embeddings to text embeddings.
3. **Training**:
   - Compute the cosine similarity between predicted and actual text embeddings.
   - Minimize the loss function (1 - cosine similarity) to optimize the model.
4. **Evaluation**:
   - Evaluate the model on a test dataset by computing the cosine similarity between predicted and actual text embeddings.
   - Save the results to a CSV file.

## Getting Started
### Prerequisites
- Python no more than 3.10.16
- PyTorch 1.13.1
- transformers 4.26.1

### Installation
1.Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Code
1.Prepare your dataset:

Place your images in the ./data/images directory.
Create a CSV file (./data/prompts.csv) with columns imgId and prompt.

2.Train the model:
```bash
python main.py
```

3.Evaluate the model:

The evaluation results will be saved to test_results.csv.

4. Generative model:

For generative model, first enter the prompt-generation folder.

(2) use download_data.py file to firstly download data

(3) use encoder_decoder.py to train on data

(4) use generate_pro_new.py to do a test on the model

### Results
The evaluation results will include the cosine similarity between the predicted text embeddings and the actual text embeddings for each image. A higher cosine similarity indicates better performance.

### There are two files too large(bert_localpath and best_model.pth), and we cannot submit them through ZIP. So we will sumbit them to instructors by google email. Please put the two files together with main.py when running the main.code