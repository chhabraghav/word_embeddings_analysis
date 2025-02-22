# Word Embeddings Analysis

## Overview
This project explores word embeddings using pre-trained GloVe vectors. It involves dimensionality reduction using Truncated SVD and visualization of word relationships.

## Features
- Load pre-trained GloVe embeddings
- Reduce dimensionality using Truncated SVD
- Visualize word relationships
- Analyze word embeddings

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage
1. **Load Word Embeddings**
   - Load pre-trained GloVe embeddings from a file.
2. **Reduce Dimensions**
   - Use Truncated SVD to reduce word embeddings to a lower-dimensional space.
3. **Plot Word Embeddings**
   - Visualize relationships between selected words.

## Example
```python
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt

def reduce_to_k_dim(embeddings, k=2):
    svd = TruncatedSVD(n_components=k, random_state=42)
    return svd.fit_transform(embeddings)
```

## Results
- Analysis of word similarities and relationships.
- Visualization of embeddings in 2D space.
