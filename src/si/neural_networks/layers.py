import numpy as np
from si.neural_networks.layers import Layer


class Dropout(Layer):
    """
    Dropout layer.

    Randomly sets a fraction of the input units to zero during training,
    helping prevent overfitting.

    Parameters
    ----------
    probability : float
        Dropout rate (between 0 and 1). Fraction of neurons to drop.
    """

    def __init__(self, probability: float):
        if probability < 0 or probability >= 1:
            raise ValueError("probability must be in the interval [0, 1)")
        self.probability = probability
        self.mask = None
        self.input = None
        self.output = None

    def forward_propagation(self, input: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward propagation for Dropout.

        Parameters
        ----------
        input : np.ndarray
            Input array.
        training : bool
            Indicates whether the network is in training mode.

        Returns
        -------
        np.ndarray
            Output of the dropout layer.
        """
        self.input = input

        # In inference mode, dropout is disabled
        if not training:
            self.output = input
            return self.output

        # Scaling factor to keep expected value constant
        scale = 1.0 / (1.0 - self.probability)

        # Binomial mask: 1 with prob (1 - p), 0 with prob p
        self.mask = np.random.binomial(
            n=1,
            p=1.0 - self.probability,
            size=input.shape
        )

        # Apply mask and scaling
        self.output = input * self.mask * scale
        return self.output

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Backward propagation for Dropout.

        Parameters
        ----------
        output_error : np.ndarray
            Gradient of the loss with respect to the layer output.

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to the layer input.
        """
        return output_error * self.mask

    def output_shape(self):
        """
        Dropout does not change the shape of the input.
        """
        return self.input_shape

    def parameters(self) -> int:
        """
        Dropout has no learnable parameters.
        """
        return 0
