import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, prediction_tensor, label_tensor):
        # Store the prediction tensor for later use in the backward pass
        self.input = prediction_tensor
        # Compute the cross-entropy loss between the prediction and label tensors
        l = -np.log(prediction_tensor + np.finfo(float).eps)
        # Set the loss to zero where the label tensor is not equal to 1
        l[label_tensor != 1] = 0
        # Compute the total loss by summing the element-wise losses
        l = l.sum()
        return l

    def backward(self, label_tensor):
        # Compute the error tensor by dividing the negative label tensor by the prediction tensor
        error_tensor = -label_tensor / (self.input + np.finfo(float).eps)
        return error_tensor
