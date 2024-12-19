# MLP

- The MLP class take in a specified number of input features (num_input), a list of hidden layers 
(hid_layers), the number of output features (num_output), and an optional activation function 
(defaulting to ReLU). In the constructor (__init__), the input layer is first defined using nn.Linear, 
and then hidden layers are iteratively added. The layers are stored in a nn.ModuleList, and the 
chosen activation function is applied between hidden layers.

- The function get_activation_function(name) is defined to return the appropriate activation function 
(ReLU, Sigmoid, Tanh, or LeakyReLU) based on a string input from the user. The forward method 
loops through all layers except the last one, applying the specified activation function at each step. 
- The user is prompted to specify the number of hidden layers, the number of neurons in each hidden 
layer, and the activation function to be used. Once all inputs are collected, the MLP model is 
instantiated with the provided parameters.

- The function get_optimizer that selects an optimizer for training a neural network based on user 
input. It supports Stochastic Gradient Descent (SGD), Adam, and Adagrad optimizers from 
PyTorch, each initialized with the model's parameters and a specified learning rate (initial_lr). If 
an unrecognized optimizer name is provided, the function defaults to using SGD. The user is 
prompted to input their choice of optimizer, which is then applied to the model.
 
- In the program, it checks if a cuda capable GPU is available and sets the device to "cuda". 
Otherwise, it defaults to the CPU. We create an instance of the MLP (Multilayer Perceptron) model 
with input parameters such as the number of input features, hidden layers, number of output labels, 
and activation function and the device.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

print(f"Model is using device: {device}") 

Instantiate the MLP model with user input 

Model = MLP(num_input, hidden_layers, num_output, activation_function=activation_function, device=device) 


- In the program PyTorch's torchvision is used to download the dataset, then creates subsets for 
training, validation, and testing, with respective sizes of 48,000, 12,000, and 10,000 samples. 
DataLoader is used to load the subsets into batches of size 20. The load_data function reshapes 
each batch of images into 1D tensors of size 784 and stores the corresponding labels. Finally, the 
training, validation, and test data are loaded into variables for model training and evaluation.


B. Fashion MNIST dataset: For a sample execution, I have used the parameters below for model 
training. 
Parameters: 
Number of hidden layers: 3 

Neurons in each layer sequentially: 150, 100, 100 

Activation function: sigmoid 

Optimization method: adam 

MLP( 
(activation_function): Sigmoid() 
(layers): ModuleList( 
(0): Linear(in_features=784, out_features=150, bias=True) 
(1): Linear(in_features=150, out_features=100, bias=True) 
(2): Linear(in_features=100, out_features=100, bias=True) 
(3): Linear(in_features=100, out_features=10, bias=True) 
) 
) 

a) Training and validation accuracy after each epoch-  
Below graph shows training and validation accuracy after each epoch. 

At epoch 0: 
Training accuracy is 60.13 and validation accuracy is 73.87 

At epoch 99: 
Training accuracy is 94.36 and validation accuracy is 88.53. 

![image](https://github.com/user-attachments/assets/255ad568-bc50-4ba3-b91d-f1b96cb1c7a5)


b) Testing accuracy- 
Testing accuracy with the best model is 87.44 

c) Confusion matrix- 

Confusion matrix is diagonally dominant, but still some of the images are wrongly classified. So 
we can assume that there are some misclassification occured.

![image](https://github.com/user-attachments/assets/c17aa292-ac18-4d0a-8a07-2c200452fcf8)



