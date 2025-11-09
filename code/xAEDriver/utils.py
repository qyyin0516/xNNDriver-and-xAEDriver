import numpy as np
import pandas as pd
import tensorflow as tf
import random
from scipy.optimize import minimize
from scipy.special import comb

tf.config.experimental_run_functions_eagerly(True)
random.seed(2)
np.random.seed(2)
tf.random.set_seed(2)


"""
Calculates the log-probability for a truncated binomial distribution (excluding k=0 and k=n).
"""
def truncated_binomial_log_pmf(k, n, p):
    if k < 1 or k > n - 1:
        return -np.Inf 
    numerator = np.log(comb(n, k)) + k * np.log(p) + (n - k) * np.log(1 - p)
    denominator = np.log(1 - (1 - p) ** n - p ** n)
    return numerator - denominator

"""
Calculates the negative log-likelihood for the given data.
"""
def log_likelihood(p, data, n):
    ll = np.sum(truncated_binomial_log_pmf(x, n, p) for x in data)
    return -ll

""" 
Estimates the probability parameter 'p' by minimizing the negative log-likelihood.
"""
def estimate_p(mutation_sum, n, initial_guess=0.8):
    result = minimize(log_likelihood, initial_guess, args=(mutation_sum, n), bounds=[(0, 1)], method='Nelder-Mead')
    estimated_p = result.x[0]
    return estimated_p

"""
Calculates relevance scores for pathway using the Layer-wise Relevance Propagation (LRP) algorithm.
"""
def calculate_relevance(X_sample, autoencoder, epsilon=0.01):
    # Forward pass to get the activation values of each layer in the network.
    X = X_sample
    activations = [tf.convert_to_tensor(X)]
    encoder_layers = autoencoder.encoder.dense_layers
    encoder_weights = autoencoder.encoder.weight_variables
    for i in range(len(encoder_layers)):
        X = encoder_layers[i](X)
        if i != len(encoder_layers) - 1:
            activations.append(X)
    X = autoencoder.encoder.regularization_layer(X)
    activations.append(X)
    decoder_layers = autoencoder.decoder.dense_layers
    decoder_weights = autoencoder.decoder.weight_variables
    for i in range(len(decoder_layers)):
        X = decoder_layers[i](X)
        activations.append(X)
    
    # Extract all weights from the model.
    weights = []
    for w in encoder_weights:
        weights.append(w.numpy())
    for w in decoder_weights:
        weights.append(w.numpy())
    
    # Backward pass to propagate relevance from the output layer back to the input layer.
    R_cur = activations[-1].numpy()
    R = [R_cur]
    for i in range(len(activations) - 2, -1, -1):
        previous_activation = activations[i].numpy()
        weight = weights[i]
    
        # Apply the LRP-epsilon rule for relevance propagation.
        z = previous_activation @ weight + epsilon
        s = R_cur / z
        c = s @ weight.T
        R_cur = previous_activation * c
        R.append(R_cur)
    R.reverse()

    # Normalize the relevance scores for each layer.
    for i in range(len(R)):
        means = np.mean(R[i], axis=1, keepdims=True)
        stds = np.std(R[i], axis=1, keepdims=True)
        stds[stds == 0] = 1 # Prevent division by zero.
        R[i] = (R[i] - means) / stds
    return R