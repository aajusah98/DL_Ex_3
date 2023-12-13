import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()      # Call the constructor of the superclass
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0, 1, (self.input_size + 1, self.output_size))
        # Randomly initialize weights with shape (input_size+1, output_size) where +1 is for the bias term
        self.trainable = True   # Indicate that this layer's parameters can be trained
        self._optimizer = None  # Initialize optimizer as None
        self._gradient_weights = None  # Initialize gradient_weights as None

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        bias = bias_initializer.initialize((1,self.output_size), 1, self.output_size)
        self.weights=np.vstack([weights, bias])

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    def forward(self, input_tensor):
        self.input_tensor = np.concatenate((input_tensor, np.ones((input_tensor.shape[0], 1))), axis=1)
        # Add a column of ones (for the bias term) to the input_tensor
        self.output_tensor = np.dot(self.input_tensor, self.weights)
        # Perform matrix multiplication between input_tensor and weights to get the output_tensor
        return self.output_tensor

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        E = np.dot(self.error_tensor, self.weights.T)
        # Calculate the error tensor of the current layer by multiplying error_tensor with the transpose of weights
        gradient_tensor = np.dot(self.input_tensor.T, self.error_tensor)
        # Calculate the gradient tensor by multiplying the transpose of input_tensor with error_tensor
        self.gradient_weights = gradient_tensor
        # Set the gradient_weights to the calculated gradient tensor
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, gradient_tensor)
            # If an optimizer is set, update the weights using the optimizer's calculate_update method
        return E[:, :-1]
        # Return E[:, :-1] to remove the column of ones added during the forward pass
