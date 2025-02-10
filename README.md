# PyTorchJacard

This is an expansion of my previous work on manipulating neuron weights but using a more typical neural network and dataset. The key differences are explained below. 

## Dataset

The dataset is a version of MNIST where we only take the 0 and 1 values from the dataset. We apply an autoencoder, which is a multilayer perceptron with 784 input neurons for MNIST. It consists of several rectified fully connected layers, each gradually reducing the dimensionality from ```784 → 512 → 256 → 128 → 64 → encoding_dim``` where ```encoding_dim``` is a user-specified variable. In our case, we reduce the dimensionality down to 30. This is so that, when we calculate the SHAP values, we have substansial enough values that we can use to manipulate the neuron weights.

## Model

This neural network is a multilayer perceptron with 30 input neurons for the encoded MNIST dataset, followed by four rectified hidden layers of 100 neurons each. Dropout is applied on the hidden layers, scaling from 0.2 to 0.5 in increments of 0.1. The output is a single neuron for binary classification with sigmoid applied. Loss is binary coressentropy with a batch size of 64. The model is trained for 25 epochs.

## SHAP Values

SHAP values are calculated using ```DeepExplainer``` class from the SHAP library. For inter-layer SHAP values, we take the trained neural net and split it before each linear layer. This effectively creates two models where the output of one becomes the input of the other. We can use to calculate SHAP value vectors compatible with the dimensionality of the hidden layers.

## Weight Manipulation

For a given cluster of SHAP values and corresponding inputs, we determine the neurons with the highest activations across all layers. Weight maniplation is done on a copy of the neural network, where increments are applied using the SHAP value vectors as a mask to determine what is changed. We can choose to modify in the direction of features that are either important (denoted as 1 in the vector) or unimportant (denoted as 0 in the vector). We calculate the corresponding accuracy and loss values to assess what impact this change has had.

## Libraries Used

The key libraries include:
* **PyTorch** - used for defining and training the neural network.
* **SHAP** - used for creating explanations for feature importance and the dataset.
* **Numpy** - used for data handling and processing.
* **Scikit Learn** - used for dimensionality reduction tasks, spliting the dataset.
* **Pandas** - used for processing and saving the output data.