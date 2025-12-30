import numpy as np
from si.neural_networks.layers import Layer


class ActivationLayer(Layer):
    """
    Base class for activation layers.

    Implements generic forward and backward propagation logic
    using an activation function and its derivative.
    """

    def __init__(self):
        self.input = None
        self.output = None

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Activation function (to be implemented by subclasses).
        """
        raise NotImplementedError

    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Derivative of the activation function (to be implemented by subclasses).
        """
        raise NotImplementedError

    def forward_propagation(self, input: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward propagation through the activation layer.
        """
        self.input = input
        self.output = self.activation_function(input)
        return self.output

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Backward propagation through the activation layer.
        """
        return output_error * self.derivative(self.input)

    def output_shape(self):
        """
        Activation layers do not change the input shape.
        """
        return self.input_shape

    def parameters(self) -> int:
        """
        Activation layers have no learnable parameters.
        """
        return 0



# Exercise 13.1 — TanhActivation

class TanhActivation(ActivationLayer):
    """
    Hyperbolic tangent activation function.

    Range: [-1, 1]
    """

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        return np.tanh(input)

    def derivative(self, input: np.ndarray) -> np.ndarray:
        t = np.tanh(input)
        return 1.0 - t ** 2


# Exercise 13.2 — SoftmaxActivation

class SoftmaxActivation(ActivationLayer):
    """
    Softmax activation function (stable version).

    Converts raw scores into probabilities that sum to 1.
    """

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        # Numerical stability trick
        shifted = input - np.max(input, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Element-wise derivative (simplified for this framework).
        """
        s = self.activation_function(input)
        return s * (1.0 - s)
