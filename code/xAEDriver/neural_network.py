import numpy as np
import tensorflow as tf
import math
import random
import copy
from utils import *

tf.config.experimental_run_functions_eagerly(True)
random.seed(2)
np.random.seed(2)
tf.random.set_seed(2)


def train_autoencoder(input_data, masking, encoder_layer_sizes, decoder_layer_sizes, n, p, batch_size, reg_type='L2', alpha_reg=0.1, alpha_binomial=0.001, num_epochs=1000, learning_rate=0.01, print_loss=False):
    """Defines, builds, and trains the autoencoder model."""

    class RegularizationLayer(tf.keras.layers.Layer):
        """A custom layer to calculate and add regularization and binomial distribution losses."""
        def __init__(self, reg_type, alpha_reg, alpha_binomial, n, p, **kwargs):
            """Initializes the layer and its loss-related parameters."""
            super(RegularizationLayer, self).__init__(**kwargs)
            self.reg_type = reg_type
            self.alpha_reg = alpha_reg
            self.alpha_binomial = alpha_binomial
            self.n = n
            self.p = p

        def call(self, inputs):
            """Calculates and adds the combined loss during the forward pass."""
            # Calculate the loss based on the truncated binomial distribution
            rounded = tf.math.round(tf.reduce_sum(inputs, axis=0))
            mapped = tf.map_fn(lambda k: truncated_binomial_log_pmf(k, self.n, self.p), rounded, dtype=tf.float32)
            distribution_loss = -tf.reduce_mean(mapped)
            
            # Calculate the standard regularization loss (L1 or L2)
            if self.reg_type == 'L2':
                regularization_loss = tf.reduce_mean(tf.square(inputs))
            elif self.reg_type == 'L1':
                regularization_loss = tf.reduce_mean(tf.abs(inputs))
            
            # Add the combined loss to the layer
            self.add_loss(self.alpha_reg * regularization_loss + self.alpha_binomial * distribution_loss)
            return inputs
    
    class Encoder(tf.keras.layers.Layer):
        """Defines the encoder part of the autoencoder."""
        def __init__(self, masking, encoder_layer_sizes):
            """Initializes the encoder layers and weights."""
            super(Encoder, self).__init__()
            self.masking = masking
            self.encoder_layer_sizes = encoder_layer_sizes
            self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.encoder_layer_sizes[0],), name='encoder_input')

            self.dense_layers = [
                tf.keras.layers.Dense(units, activation=None, name=f'encoder_dense_{i}') 
                for i, units in enumerate(self.encoder_layer_sizes[1:])
            ]
            self.regularization_layer = RegularizationLayer(reg_type=reg_type, alpha_reg=alpha_reg, alpha_binomial=alpha_binomial, n=batch_size, p=p, name='encoder_regularization')
            self.weight_variables = [tf.Variable(tf.random.normal([self.encoder_layer_sizes[i - 1], self.encoder_layer_sizes[i]])) for
                                     i in range(1, len(self.encoder_layer_sizes))]
            self.bias_variables = [tf.Variable(tf.random.normal([self.encoder_layer_sizes[i]])) for i in
                                   range(1, len(self.encoder_layer_sizes))]

        def call(self, inputs):
            """Defines the forward pass for the encoder."""
            x = self.input_layer(inputs)
            for i, layer in enumerate(self.dense_layers):
                if i != len(encoder_layer_sizes) - 2:
                    x = tf.nn.tanh(tf.add(tf.matmul(x, self.weight_variables[i]), self.bias_variables[i]))
                else:
                    x = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weight_variables[i]), self.bias_variables[i]))
            x = self.regularization_layer(x)
            return x
    
    
    class Decoder(tf.keras.layers.Layer):
        """Defines the decoder part of the autoencoder."""
        def __init__(self, masking, decoder_layer_sizes):
            """Initializes the decoder layers and weights."""
            super(Decoder, self).__init__()
            self.masking = masking
            self.decoder_layer_sizes = decoder_layer_sizes
            self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.decoder_layer_sizes[0],), name='decoder_input')
            self.dense_layers = [
                tf.keras.layers.Dense(units, activation=None, name=f'decoder_dense_{i}') 
                for i, units in enumerate(self.decoder_layer_sizes[1:])
            ]
            self.weight_variables = [tf.Variable(tf.random.normal([self.decoder_layer_sizes[i], self.decoder_layer_sizes[i + 1]])) for
                                     i in range(len(self.decoder_layer_sizes) - 1)]
            self.bias_variables = [tf.Variable(tf.random.normal([self.decoder_layer_sizes[i + 1]])) for i in
                                   range(len(self.decoder_layer_sizes) - 1)]
    
        def call(self, inputs):
            """Defines the forward pass for the decoder, applying pathway masks."""
            x = inputs
            for i, layer in enumerate(self.dense_layers):
                if i == 0:
                    x = tf.nn.tanh(tf.add(tf.matmul(x, self.weight_variables[i]), self.bias_variables[i]))
                else:
                    # Apply the pathway-based mask to the weights
                    masked_weights = tf.multiply(self.weight_variables[i], self.masking[i - 1])
                    x = tf.nn.tanh(tf.add(tf.matmul(x, masked_weights), self.bias_variables[i]))
            return x

    
    class Autoencoder(tf.keras.Model):
        """A full autoencoder model combining the encoder and decoder."""
        def __init__(self, encoder, decoder, masking):
            """Initializes the autoencoder model."""
            super(Autoencoder, self).__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.masking = masking
    
        def call(self, inputs):
            """Defines the full forward pass from input to reconstruction."""
            encoded = self.encoder(inputs)
            decoded = self.decoder(encoded)
            return decoded
    
    
    # Model and Optimizer Initialization
    encoder = Encoder(masking, encoder_layer_sizes)
    decoder = Decoder(masking, decoder_layer_sizes)
    autoencoder = Autoencoder(encoder, decoder, masking)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    
    @tf.function
    def train_step(inputs):
        """Executes a single training step."""
        with tf.GradientTape() as tape:
            # Get model predictions and calculate losses
            predictions = autoencoder(inputs)
            reconstruction_loss = tf.keras.losses.mean_squared_error(inputs, predictions)
            # Fetch regularization losses from the custom layer
            regularization_loss = sum(autoencoder.encoder.losses) 
            total_loss = reconstruction_loss + regularization_loss
        
        # Calculate and apply gradients
        gradients = tape.gradient(total_loss, autoencoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
        return tf.reduce_mean(total_loss).numpy()
    
    
    # Model Training Loop 
    for epoch in range(num_epochs):
        # Store weights from the previous epoch
        if epoch != 0:
            prev_weight = copy.deepcopy(autoencoder.weights)
            
        num_samples = len(input_data)
        num_batches = math.ceil(num_samples / batch_size)
    
        total_loss = 0.
        # Iterate over mini-batches
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch_x = input_data[start_idx:end_idx]
            
            loss = train_step(batch_x)
            total_loss += loss
        
        avg_loss = total_loss / num_batches
        if avg_loss == np.Inf:
            break
        
        if print_loss:
            print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_loss))

    # At the end of all epochs, restore the weights from the last epoch without Inf
    if 'prev_weight' in locals() and num_epochs > 1:
        autoencoder.set_weights(prev_weight)
        
    return autoencoder