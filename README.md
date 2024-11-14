# MNIST Digit Classification using Naive Bayes with Dirichlet Priors

This project implements a Naive Bayes classifier with symmetric Dirichlet priors for the MNIST digit dataset. The classifier uses Bayesian inference to predict handwritten digit classes (0–9) based on pixel values, leveraging Dirichlet priors to improve class probability estimation and robustness, particularly when dealing with sparse or imbalanced data.

## Project Overview

The classifier is trained on a preprocessed version of the MNIST dataset, where each image is represented as a 28×28 pixel matrix with values ranging from 0 to 255. The Naive Bayes algorithm is enhanced with symmetric Dirichlet priors, which adjust the probability estimations by applying pseudo-counts to the class probability estimates, improving performance in this context.

The key elements include:
- **Dirichlet Prior Adjustment**: Utilizes pseudo-counts `α` to refine the class probability distribution and class-conditional posteriors.
- **Class Probability Estimation**: Calculates probabilities based on counts adjusted by the Dirichlet prior, where the prior smooths the probability estimates.

## Dependencies

- `numpy`
- `scikit-learn`
- `matplotlib`

Install dependencies with:
```bash
pip install numpy scikit-learn matplotlib
