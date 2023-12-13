from copy import deepcopy


class NeuralNetwork:
    def __init__(self, optimizer,weights_initializer, bias_initializer) -> None:
        self.optimizer = optimizer
        self.loss = []  # List to store the loss values during training
        self.layers = []  # List to store the layers of the neural network
        self.data_layer = None  # Reference to the data layer
        self.loss_layer = None  # Reference to the loss layer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def forward(self):
        # Perform forward pass through the network
        inp, op = self.data_layer.next()  # Get input data and labels from the data layer
        self.label = op  # Store the labels for later use in the backward pass
        for layer in self.layers:
            inp = layer.forward(inp)  # Forward pass through each layer
        inp = self.loss_layer.forward(inp, self.label)  # Compute the final output and loss
        self.pred = inp  # Store the predicted output
        return self.pred

    def backward(self):
        # Perform backward pass through the network
        loss = self.loss_layer.backward(self.label)  # Compute the error tensor from the loss layer
        for layer in self.layers[::-1]:
            loss = layer.backward(loss)  # Backward pass through each layer, updating gradients

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)  # Assign a copy of the optimizer to the layer if it is trainable
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)  # Add the layer to the list of layers in the network

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()  # Perform forward pass
            self.backward()  # Perform backward pass and update gradients
            self.loss.append(loss)  # Store the loss value

    def test(self, input_tensor):
        inp = input_tensor  # Get the input tensor
        for layer in self.layers:
            inp = layer.forward(inp)  # Perform forward pass through each layer
        return inp  # Return the output tensor

