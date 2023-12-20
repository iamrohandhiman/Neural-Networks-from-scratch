# Neural Networks from Scratch
## - Rohan Dhiman

### Index:
- Create the Base Layer
- Create the Dense Layer
- Create Activation Layer
- Implement Activation Functions and Loss Functions

### Steps:
1. Feed input, Data flows from layer to layer. Retrieve output.
   
   _y = network(x, w, b)_
2. Calculate the error
   
   _E = 1/2 * (y_hat - y)^2_
3. Adjust the parameters using gradient descent
   

   $\theta_j = \theta_j - \alpha \frac{\partial \textit{J}(\theta)}{\partial \theta_j}$

5. Start again

### Implementation Design:

![image](https://github.com/iamrohandhiman/Neural-Networks-from-scratch/assets/138308928/645065a2-8f1c-42cd-817e-3abacfc17104)

_{Error = f(Y)} → {Y = f(WX + B)}_
- _X: The input vector from the previous layer or the input layer._
- _W: The weight matrix._
- _B: The bias vector._
- _f: The activation function._
- _Y: The output of the dense layer._

Given that the input f(Y) for the error function is a composite function, we employ the Back propagation algorithm to analyze how changes in each parameter, contributing to the composite function, influence the error. Back propagation enables a systematic examination of the impact of parameter adjustments on the overall error, facilitating the iterative refinement of the model during the training process.

When updating parameters/gradient descent
$$\theta_j = \theta_j - \alpha \frac{\partial \textit{J}(\theta)}{\partial \theta_j}$$
Where,  $$\theta_j \$$ is the parameter.

In order to understand how changes in the parameters contribute to the overall error, we must calculate their partial derivatives with respect to the error $\\frac{\partial E}{\partial W} \quad \text{and} \quad \frac{\partial E}{\partial B}\$
. Since each layer in the neural network has parameters influencing the error, it is crucial to compute these derivatives for every layer.

To compute these derivatives for a given layer, we employ the chain rule. The process for each layer can be broken down as follows:

![image](https://github.com/iamrohandhiman/Neural-Networks-from-scratch/assets/138308928/b3091b80-30ce-4e5e-bdf2-460c33f7d50e)

$\\frac{\partial E}{\partial W} = \frac{\partial E}{\partial Y} \frac{\partial Y}{\partial W}\$ ,  $\\frac{\partial E}{\partial B} = \frac{\partial E}{\partial Y} \frac{\partial Y}{\partial B}\$

To uncover how the input from the previous layer impacts the error, we calculate $\\frac{\partial E}{\partial Y}\$ . Now except for the final f(Y) every other $\\frac{\partial E}{\partial Y_i} = \frac{\partial E}{\partial X_{i+1}}\$. This relationship is illustrated in the diagram below.

![image](https://github.com/iamrohandhiman/Neural-Networks-from-scratch/assets/138308928/9fa6202a-f819-463d-9955-7f5ea06d54df)

Hence, we identify that in order to update parameters and to minimize the Error, we must find
$\\frac{\partial E}{\partial Y} \quad \text{or} \quad \frac{\partial E}{\partial X}, \frac{\partial E}{\partial W}, \frac{\partial E}{\partial B}\$ for every layer. – (1)

It is crucial to emphasize that each neuron in a neural network generates a single-valued output, irrespective of the dimensionality of the input vector _X_. The size of the input vector does not impact the fact that each neuron produces a scalar output.

![image](https://github.com/iamrohandhiman/Neural-Networks-from-scratch/assets/138308928/7539d5ad-9f7f-4077-b195-4dce20fb17d8)

Let’s consider a neural network and understand forward propagation

![image](https://github.com/iamrohandhiman/Neural-Networks-from-scratch/assets/138308928/f523133d-3545-4627-9c27-7cb2e0046ae3)

X = [x₁]
    [x₂]
    [⋮ ]
    [xᵢ]


While understanding the dimensions of the input and output sizes, we gain insights into how they influence the sizing of the parameter matrix W and bias vector _B_. For _W_, the number of rows is determined by the number of neurons present in the current layer, while the number of columns corresponds to the number of neurons in the input layer or the rows of the input matrix. This substantiates the convention of using weights with subscripts _'ji'_

__Finding $\\frac{\partial E}{\partial Y} \quad \text{or} \quad \frac{\partial E}{\partial X}, \frac{\partial E}{\partial W}, \frac{\partial E}{\partial B}\$ for every layer (generalized)__



$\\frac{\partial E}{\partial W} = \frac{\partial E}{\partial Y} \frac{\partial Y}{\partial W}\$ , $\\frac{\partial E}{\partial B} = \frac{\partial E}{\partial Y} \frac{\partial Y}{\partial B}\$


Now Error is a function of Y = [y₁, y₂, ..., yⱼ] and we are trying to find how changes in these values influence the Error function, hence ∂E/∂Y = [∂E/(∂y_1), ∂E/(∂y_2), ..., ∂E/(∂y_j)].

Further, we can try to find how changing the value of W = [w_11, w_12, ..., w_1i; w_21, w_22, ..., w_2i; ..., w_j1, w_j2, ..., w_ji] influences the Error function, hence

The final equations for the backpropagation are:
1.  $\\frac{\partial E}{\partial W} = \frac{\partial E}{\partial Y} \cdot X^T\$

2.  $\\frac{\partial E}{\partial B} = \frac{\partial E}{\partial Y}\$

3.  $\\frac{\partial E}{\partial X} = W^T \cdot \frac{\partial E}{\partial Y}\$
