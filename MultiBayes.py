import numpy as np
import pandas as pd

class MultiBayes():
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.gaussian_params = {}
        self.bernoulli_params = {}
        self.categorical_params = {}

    def fit(self, X_continuous=None, X_binary=None, X_categorical=None, y=None):
        # Calculate priors P(C)
        self.classes, counts = np.unique(y, return_counts=True)
        self.class_priors = {cls: count / len(y) for cls, count in zip(self.classes, counts)}

        # Gaussian Naive Bayes for continuous features
        if X_continuous is not None:
            for cls in self.classes:
                X_class = X_continuous[y == cls]
                # Calculate mean, variance, and standard deviation for each class
                var = X_class.var(axis=0)
                self.gaussian_params[cls] = {
                    'mean': X_class.mean(axis=0),
                    'var': var,
                    'std': np.sqrt(var)
                }

        # Bernoulli Naive Bayes for binary features
        if X_binary is not None:
            for cls in self.classes:
                cls_data = X_binary[y == cls]
                self.bernoulli_params[cls] = cls_data.mean(axis=0)

        # Categorical Naive Bayes for categorical features
        if X_categorical is not None:
            for cls in self.classes:
                cls_data = X_categorical[y == cls]
                self.categorical_params[cls] = [
                    col.value_counts(normalize=True).sort_index() for _, col in cls_data.items()
                    # Use .items() instead of .iteritems()
                ]

    def calculate_likelihood(self, mean, var, x):
        # Likelihood calculation based on Gaussian distribution
        eps = 1e-6  # Small number to prevent division by zero
        coef = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
        exp = np.exp(-(x - mean) ** 2 / (2.0 * var + eps))
        return coef * exp

    def predict(self, X_continuous=None, X_binary=None, X_categorical=None):
        posteriors = []

        for cls in self.classes:
            log_prob = np.log(self.class_priors[cls])

            # Gaussian likelihood for continuous features
            if X_continuous is not None:
                mean = self.gaussian_params[cls]['mean']
                var = self.gaussian_params[cls]['var']
                log_prob += np.sum(-0.5 * np.log(2 * np.pi * var) - (X_continuous - mean) ** 2 / (2 * var), axis=1)

            # Bernoulli likelihood for binary features
            if X_binary is not None:
                bernoulli_prob = self.bernoulli_params[cls]
                log_prob += np.sum(X_binary * np.log(bernoulli_prob) + (1 - X_binary) * np.log(1 - bernoulli_prob), axis=1)

            # Categorical likelihood for categorical features
            if X_categorical is not None:
                for i, prob in enumerate(self.categorical_params[cls]):
                    col = X_categorical.iloc[:, i]
                    log_prob += col.map(lambda x: np.log(prob.get(x, 1e-6)))

            posteriors.append(log_prob)

        # Convert to DataFrame for easier manipulation
        posteriors = np.array(posteriors).T  # Transpose to match rows with classes
        return self.classes[np.argmax(posteriors, axis=1)]
