
# Neural Network with Custom Implementation in Python

A Simple Neural Network Library for Multi-Layer Perceptron (MLP) using NumPy 

This project stems from my curiosity and passion for translating mathematical equations into practical applications. Inspired by resources such as The Independent Code, Andrew Ng's Machine Learning course, StatQuest, and 3Blue1Brown. The implementation utilizes NumPy , providing a straightforward look into the workings of a multi-layer perceptron. 

check out math here :   [Neural Network Math by Rohan Dhiman](https://drive.google.com/file/d/1gCDmMrSBulgsnRXG4yBMVx0m72YoE9BG/view?usp=sharing)
#

* **Import Libraries:** Import `NumPy`, `Keras` for `MNIST`, and `Matplotlib`.

* **Define `Dense` Class:** Custom class for a fully connected layer with random weights and biases.

* **Define `Activation` Class:**  Encapsulation of activation functions for forward and backward propagation.

* **Implement `Tanh` Activation:** Subclass of Activation for the tanh activation function.

* **Define Loss Functions:** MSE and its derivative for error calculation.

* **Implement `Predict` Function:**  Function for predicting output by `forward propagation` through network layers.

* **Implement Training Function:** Train the neural network by iterating epochs, calculating error, and updating weights`gradient descent`.

* **Preprocess Data:**  `Reshape`, normalize, and convert labels for data preprocessing.

* **Loading and Preprocessing MNIST Dataset:** Load and preprocess `MNIST` training and testing data.

* **Initialize and Train Neural Network:** Create and train the `neural network` with specified layers.

* **Test and Visualize Results:** Iterate through the test dataset, make predictions, and visualize with Matplotlib.
# Dense Layer
## `Dense` Class
### Initialization
```python
class Dense():
    def __init__(self, i, j):
        """
        Initialize the Dense layer with random weights and biases.

        Parameters:
        - i (int): Number of input neurons.
        - j (int): Number of output neurons.

        Attributes:
        - weights (numpy array): Randomly initialized weights.
        - biases (numpy array): Randomly initialized biases.
        """
        self.weights = np.random.randn(j, i)
        self.biases = np.random.randn(j, 1)
```

The `Dense` class is initialized with two parameters `i` and `j`, representing the number of input and output neurons, respectively. It initializes weights and biases with random values using NumPy's `randn` function.

### Forward Propagation

```python
class Dense():
   def forward_propagation(self, inpx):
    """
    Perform forward propagation through the Dense layer.

    Parameters:
    - inpx (numpy array): Input to the Dense layer.

    Returns:
    - numpy array: Output of the Dense layer after   applying     weights, biases, and activation.
    """
    self.inpx = inpx
    return np.dot(self.weights, self.inpx) + self.biases

```
The `forward_propagation` method takes an input (`inpx`) and computes the output by performing a dot product of `weights and input, adding biases .

### Backward Propagation

```python
def backward_propagation(self, DelE_DelY, alpha):
    """
    Perform backward propagation through the Dense layer.

    Parameters:
    - DelE_DelY (numpy array): Gradient of the error with respect to the output.
    - alpha (float): Learning rate for weight and bias updates.

    Returns:
    - numpy array: Gradient of the error with respect to the input.
    """
    DelE_DelW = np.matmul(DelE_DelY, self.inpx.T)
    DelE_DelX = np.matmul(self.weights.T, DelE_DelY)
    self.weights -= alpha * DelE_DelW
    self.biases -= alpha * DelE_DelY
    return DelE_DelX

```
For `backward_propagation`, it takes the derivative of the error with respect to the output (`DelE_DelY`) and the learning rate (`alpha`). It calculates the derivative of the error with respect to the weights (`DelE_DelW`) and the input (`DelE_DelX`), updates the weights and biases, and returns the derivative of the error with respect to the input.

# Activations

The `Activation` class in the provided code serves as a foundation for implementing activation functions in neural networks. It is designed to encapsulate both the activation function and its derivative for forward and backward propagation, respectively.

## `Activation` Class

### Initialization

```python
def __init__(self, act, act_prime):
    """
    Initialize the Activation class with activation functions.

    Parameters:
    - act (function): Activation function.
    - act_prime (function): Derivative of the activation function.

    Attributes:
    - act (function): Activation function.
    - prime (function): Derivative of the activation function.
    """
    self.act = act
    self.prime = act_prime

```
The `Activation` class is initialized with two functions: `act` representing the activation function and `act_prime` representing its derivative.

### Forward Propagation
```python
def forward_propagation(self, inpx):
    """
    Perform forward propagation through the Activation layer.

    Parameters:
    - inpx (numpy array): Input to the Activation layer.

    Returns:
    - numpy array: Output of the Activation layer after applying the activation function.
    """
    self.inpx = inpx
    return self.act(self.inpx)
```
The `forward_propagation` method takes an input (`inpx`) and computes the output using the specified activation function.

### Backward Propagation
```python
def backward_propagation(self, DelE_DelY, alpha):
    """
    Perform backward propagation through the Activation layer.

    Parameters:
    - DelE_DelY (numpy array): Gradient of the error with respect to the output.
    - alpha (float): Learning rate for weight and bias updates.

    Returns:
    - numpy array: Gradient of the error with respect to the input.
    """
```
    return DelE_DelY * self.prime(self.inpx)

### Tanh Class (Subclass of Activation)
The `Tanh` class is a specific implementation of the Activation class, using the hyperbolic tangent (tanh) activation function.

```python 
def __init__(self):
    """
    Initialize the Tanh class as a specific implementation of the Activation class with tanh activation.

    Attributes:
    - act (function): Tanh activation function.
    - prime (function): Derivative of the tanh activation function.
    """
    def tanh(x):
        return np.tanh(x)

    def tanh_prime(x):
        return 1 - np.tanh(x) ** 2

    super().__init__(tanh, tanh_prime)
```
In the `Tanh` class, the `__init__` method initializes the superclass (`Activation`) with the tanh activation function and its derivative.

# Loss Function : Mean Squared Error (MSE) 

The provided code defines two functions related to Mean Squared Error (MSE). This documentation outlines the purpose and usage of the `mse` and `mse_prime` functions.

### `mse` Function
```python
def mse(y_true, y_pred):
    """
    Calculate the mean squared error between true values and predicted values.

    Parameters:
    - y_true (numpy array): True values.
    - y_pred (numpy array): Predicted values.

    Returns:
    - float: Mean squared error.
    """
    return np.sum(np.mean((y_pred - y_true) ** 2))
```

The `mse` function calculates the mean squared error between the true values (`y_true`) and the predicted values (`y_pred`). It computes the squared difference between each corresponding pair of true and predicted values, takes the mean across all pairs, and then sums up these mean values. The function returns a single float value representing the mean squared error.

### `mse_prime` Function
```python
def mse_prime(y_true, y_pred):
    """
    Calculate the derivative of mean squared error with respect to predicted values.
    
    Parameters:
    - y_true (numpy array): True values.
    - y_pred (numpy array): Predicted values.

    Returns:
    - numpy array: Output gradient for the last layer.
    """
    return 2 * (y_pred - y_true) / y_true.shape[0]
```

The `mse_prime` function calculates the derivative of mean squared error with respect to the predicted values (`y_pred`). It returns a numpy array representing the output gradient for the last layer of a neural network. This gradient is used during the backpropagation process for updating the parameters of the last layer.


# Data Preprocessing
```python
def preprocess_data(x, y, limit):
    """
    Preprocess input data and labels for training or testing.

    Parameters:
    - x (numpy array): Input data.
    - y (numpy array): Labels.
    - limit (int): Limit on the number of samples to preprocess.

    Returns:
    - Tuple (x_processed, y_processed): Processed input data and labels.
    ```
    x = x.reshape(x.shape[0], 784, 1)  # Reshape input data to (number of samples, 784, 1)
    x = x.astype("float32") / 255  # Normalize input data to the range [0, 1]
    y = to_categorical(y)  # Convert labels to one-hot encoding
    y = y.reshape(y.shape[0], 10, 1)  # Reshape labels to (number of samples, 10, 1)

    return x[:limit], y[:limit]
```

This function preprocesses input data (`x`) and labels (`y`) for training or testing. It reshapes the input data, normalizes it, converts labels to one-hot encoding, and returns a tuple containing the processed input data (`x_processed`) and labels (`y_processed`).

### Example Usage:

```python
x_processed, y_processed = preprocess_data(x_train, y_train, 5000)
```
# Predict
```python
def predict(network, inpx):
    """
    Perform forward propagation through the entire neural network to make predictions.

    Parameters:
    - network (list): List of layers forming the neural network.
    - inpx (numpy array): Input data for prediction.

    Returns:
    - numpy array: Output prediction of the neural network.
    """
    output = inpx
    for layer in network:
        output = layer.forward_propagation(output)
    return output
```

The `predict` function takes a neural network (`network`) and an input dataset (`inpx`). It iterates through each layer in the neural network, applying forward propagation to compute the final output prediction. The function returns the output prediction as a NumPy array.

# Train 
```pyhton
def train(network, loss, loss_prime, x_train, y_train, count=100, learning_rate=0.1):
    """
    Train the neural network using backpropagation.

    Parameters:
    - network (list): List of layers forming the neural network.
    - loss (function): Loss function used for training.
    - loss_prime (function): Derivative of the loss function.
    - x_train (numpy array): Input training data.
    - y_train (numpy array): True labels for training data.
    - count (int): Number of training epochs (default is 100).
    - learning_rate (float): Learning rate for weight and bias updates (default is 0.1).

    """
    for e in range(count):
        error = 0
        for x, y in zip(x_train, y_train):
            output = predict(network, x)
            error += loss(y, output)
            grad = loss_prime(y, output)

            for layer in reversed(network):
                grad = layer.backward_propagation(grad, learning_rate)
        print(f"{e + 1}/{count}, error={error}")
```
The `train` function is responsible for training a neural network through a specified number of epochs (`count`) using backpropagation. It takes the neural network layers (`network`), a loss function (`loss`), the derivative of the loss function (`loss_prime`), input training data (`x_train`), true labels (`y_train`), and optional parameters for the number of epochs and learning rate. The function iteratively updates the network's weights and biases to minimize the error between predicted and true values.

### Example Usage:

```python
train(my_neural_network, mse, mse_prime, training_data, true_labels, count=200, learning_rate=0.01)
```

# Prediction

```python
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print('Predicted Label:', np.argmax(output))
```
This code snippet iterates through the test dataset (`x_test` and `y_test`), makes predictions using the predict function with the specified neural network (`network`), and prints the predicted label. The `np.argmax(output)` is used to find the index of the maximum value in the output array, representing the predicted label.

### Example Usage:

```python
for x, y in zip(x_test, y_test):
    output = predict(my_neural_network, x)
    print('Predicted Label:', np.argmax(output))
```
### Visualization of Test Results
```python
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.title('Actual Label: {}'.format(np.argmax(y)))
plt.show()
```
### Example Usage:
```python
# Assuming 'x' is an image from the test dataset and 'y' is the corresponding true label.
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.title('Actual Label: {}'.format(np.argmax(y)))
plt.show()
```





## Authors

- [@RohanDhiman](-www.linkedin.com/in/rohan-dhiman-867a32273)
- [@RaginiGaggar](https://www.linkedin.com/in/ragini-gaggar-978914227/)


