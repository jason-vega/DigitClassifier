# Digit Classifier
A multi-layer Java neural network featuring the stochastic gradient descent algorithm to recognize digits in the MNIST data set. The default neural network in *Train.java* has the following hyperparameters, which can be altered (as shown in the sections below):
* Layers: 784-70-35-10
* Mini-batch size: 256
* Learning rate: 0.1667
* Epochs: 200
* Cost function: Cross-Entropy
* Activation function: Sigmoid

The default neural network yielded a maximum test accuracy rate of 93.15% on the 192nd epoch when trained and tested on the entire MNIST data set.
## Prerequisites
You will need to download the MNIST data set from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/index.html). Download all four files and save them into a directory of your choice.
## Installation
Clone or download the project from GitHub. In *Train.java*, locate the following lines:
```
public static final String TRAINING_IMAGE_FILE_PATH = "";
public static final String TRAINING_LABEL_FILE_PATH = "";
public static final String TEST_IMAGE_FILE_PATH = "";
public static final String TEST_IMAGE_LABEL_PATH = "";
```
Copy the absolute path of each MNIST data set files into their appropriate string. Lastly, compile the Java program.
```
javac *.java
```
## Running the Program
Once compiled, you may train the neural network on the MNIST data set:
```
java Train
```
## Modifying Hyperparameters
There are a number of hyperparameters you can change with relative ease. All hyperparameters are stored as constants at the top of the Train class in *Train.java*. For instance, to change the learning rate to 0.01, locate the following line:
```
public static final double LEARNING_RATE = 0.1667;
```
and change it to
```
public static final double LEARNING_RATE = 0.01;
```
Recompile and run the program so that the new hyperparameter(s) can come into effect.
## Adding/Remove Layers
Locate the following lines in *Train.java*:
```
public static final int INPUT_LAYER_SIZE = 784;
public static final int FIRST_HIDDEN_LAYER_SIZE = 70;
public static final int SECOND_HIDDEN_LAYER_SIZE = 35;
public static final int OUTPUT_LAYER_SIZE = 10;
```
You can modify or remove these constants as you wish, or even add new constants for additional layers. For example, let's change the number of neurons in the second hidden layer to 40 and add a third hidden layer with 20 neurons:
```
public static final int INPUT_LAYER_SIZE = 784;
public static final int FIRST_HIDDEN_LAYER_SIZE = 70;
public static final int SECOND_HIDDEN_LAYER_SIZE = 40;
public static final int THIRD_HIDDEN_LAYER_SIZE = 20;
public static final int OUTPUT_LAYER_SIZE = 10;
```
Next, locate the following lines:
```
NeuralNetwork n = new NeuralNetwork(new int[]{
  INPUT_LAYER_SIZE,
  FIRST_HIDDEN_LAYER_SIZE,
  SECOND_HIDDEN_LAYER_SIZE,
  OUTPUT_LAYER_SIZE
});
```
The NeuralNetwork constructor accepts an array of integers denoting the number of neurons in each layer, where the *i*th integer corresponds to the *i*th layer. If you have removed any of the layer constants, added new ones or renamed them, these changes must be reflected here. Following up with our example, we change these lines to the following:
```
NeuralNetwork n = new NeuralNetwork(new int[]{
  INPUT_LAYER_SIZE,
  FIRST_HIDDEN_LAYER_SIZE,
  SECOND_HIDDEN_LAYER_SIZE,
  THIRD_HIDDEN_LAYER_SIZE,
  OUTPUT_LAYER_SIZE
});
```
We have now added a third hidden layer of 20 neurons, while the second hidden layer's neurons have increased to 40! Recompile and run the program so that the new network structure can come into effect.
## Limiting the Number of Training/Test Images
If you wish to test a smaller subset of the MNIST data set, you can limit the size of the subset by locating and modifying the following lines in *Train.java*:
```
public static final int MAX_TRAINING_INPUTS = 60000;
public static final int MAX_TEST_INPUTS = 10000;
```
For instance, say we only wish to train our neural network on the first 10,000 training images and test using the first 5,000 test images. We then shall change the above lines to the following:
```
public static final int MAX_TRAINING_INPUTS = 10000;
public static final int MAX_TEST_INPUTS = 5000;
```
Recompile and run the program to load the subset of MNIST data.
## Output Data Loading and Training Progress
Sometimes the amount of data you wish to process can make it seem like your program is indefinitely stuck. This is especially true when loading the entire MNIST training and test image sets, or training with multiple dense layers. In such cases, you may wish to output current progress info - the current block being processed for data loading, and the current mini-batch being evaluated for training - i.e. we wish to set verbose mode to *true*. Verbose mode is on by default by for loading image data, off by default for loading label data and off by default for training. To turn off verbose mode for loading image data, add a new constant variable in *Train.java*:
```
public static final int LOAD_IMAGE_VERBOSE = false;
```
Locate the following line:
```
LoadData trainImageLoad = new LoadData(TRAINING_IMAGE_FILE_PATH, 
  IMAGE_FILE_OFFSET, INPUT_LAYER_SIZE, MAX_TRAINING_INPUTS);
```
and add the new constant variable as the final parameter in the LoadData constructor:
```
LoadData trainImageLoad = new LoadData(TRAINING_IMAGE_FILE_PATH, 
  IMAGE_FILE_OFFSET, INPUT_LAYER_SIZE, MAX_TRAINING_INPUTS, LOAD_IMAGE_VERBOSE);
```
Do the same for testImageLoad if desired. To turn verbose mode on for loading label data or for training, simply locate the following lines in *Train.java*:
```
public static final boolean LOAD_LABEL_VERBOSE = false;
public static final boolean TRAIN_VERBOSE = false;
```
Set these variables to *true* as desired. Lastly, for all changes to verbose mode configuration, recompile and run the program so that the changes can take effect.
## Changing the cost function
There is no explicit cost function defined in *NeuralNetwork.java*. That's because backpropogation only concerns itself with the *derivative* of the cost function! Specifically, we define a method that represents the cost derivative with respect to the activation function called costDerivativeWithRespectToActivation(). By default, the assumed cost function is the cross-entropy function. To implement a different cost function, find an expression for the cost derivative with respect to the activation function and implement this in the aforementioned method.
## Changing the activation function
Locate the activation and activationPrime methods in *NeuralNetwork.java*. By default the activation function is the sigmoid function. Modify these methods to implement an alternative activation function (and its derivative).
## Special Thanks
This project was inspired by Michael Nielsen's online book [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/) and [3Blue1Brown's series on deep learning](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi).
