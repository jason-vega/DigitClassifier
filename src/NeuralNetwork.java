import java.util.Random;

/**
 * A representation of a neural network.
 * 
 * @author Jason Vega
 *
 */
public class NeuralNetwork {
	public static final int COST_GRADIENT_COMPONENTS = 2;
	public static final int PERCENTAGE_RATIO = 100;
	
	private int[] layers;
	private int numberOfLayers;
	private double[][][] weights;
	private double[][][] biases;
	
	/**
	 * Constructs a new neural network representation where the number of nodes
	 * in each layer is specified using the layers array.
	 * 
	 * @param layers An array representing the number of nodes in each layer.
	 */
	public NeuralNetwork(int[] layers) {
		this.layers = layers;
		this.numberOfLayers = layers.length;
		
		// Initialize random weight matrices and bias vectors
		this.weights = randomWeights(layers);
		this.biases = randomBiases(layers);
	}
	
	/**
	 * Trains this neural network on the provided training data for the 
	 * specified number of epochs.
	 * 
	 * @param trainingData The training data, including the input and 
	 * corresponding labels.
	 * @param learningRate The learning rate for the optimization method.
	 * @param epochs The number of training sessions.
	 * @param testData The test data, including the input and corresponding
	 * labels.
	 * @param verbose Whether or not to output processing data.
	 * @throws Exception 
	 */
	public void train(double[][][][] trainingData, int miniBatchSize, 
			double learningRate, int epochs, double[][][][] testData, 
			boolean verbose) 
					throws Exception {
		double weights[][][] = this.getWeights();
		double biases[][][] = this.getBiases();
		int numberOfLayers = this.getNumberOfLayers();
		
		for (int i = 1; i <= epochs; i++) {
			double[][][][][] miniBatches = this.getMiniBatches(trainingData, 
					miniBatchSize);
			
			System.out.println("-- EPOCH " + i + "/" + epochs + " --");
			
			for (int j = 0; j < miniBatches.length; j++) {
				double[][][][] miniBatch = miniBatches[j];
				double[][][][] gradientSum = 
					this.stochasticGradientDescent(miniBatch);
				double[][][] gradientSumWithRespectToWeights = gradientSum[0];
				double[][][] gradientSumWithRespectToBiases = gradientSum[1];
				
				if (verbose) {
					System.out.println("Processing mini-batch " + (j + 1) + "/" + 
							miniBatches.length);
				}
				
				for (int k = 0; k < numberOfLayers - 1; k++) {
					weights[k] = Matrix.subtract(
						weights[k], 
						Matrix.scale(learningRate / miniBatch.length, 
							gradientSumWithRespectToWeights[k])
					);
					
					biases[k] = Matrix.subtract(
						biases[k], 
						Matrix.scale(learningRate / miniBatch.length, 
							gradientSumWithRespectToBiases[k])
					);
				}
			}
			
			System.out.println("TRAINING ACCURACY: " + 
					this.getAccuracy(trainingData) + "%");
			System.out.println("TEST ACCURACY: " + this.getAccuracy(testData) + 
				"%\n");
		}
	}
	
	/**
	 * Calculates and returns the accuracy (%) of this neural network.
	 * 
	 * @param dataSet The data to test for accuracy.
	 * @return The accuracy rate as a percentage.
	 * @throws Exception 
	 */
	public double getAccuracy(double[][][][] dataSet) throws Exception {
		int correct = 0;
		
		for (double data[][][] : dataSet) {
			double[][] input = data[0];
			double[][] label = data[1];
			double[][] output = forwardPass(input);
			int resultIndex = Matrix.argmax(output);
			
			if (label[resultIndex][0] == 1) {
				correct++;
			}
		}
		
		return (double) correct / dataSet.length * PERCENTAGE_RATIO;
	}
	
	/**
	 * Perform a single forward pass through the network with the given input.
	 * 
	 * @param input The input to this neural network.
	 * @return The output of the neural network.
	 * @throws Exception 
	 */
	public double[][] forwardPass(double[][] input) throws Exception {
		double[][][] weights = this.getWeights();
		double[][][] biases = this.getBiases();
		double output[][] = input;
		
		for (int l = 0; l < this.getNumberOfLayers() - 1; l++) {
			double[][] weightMatrix = weights[l];
			double[][] biasVector = biases[l];
			
			output = Matrix.activation(
				Matrix.add(
					Matrix.product(weightMatrix, output), 
					biasVector
				)
			);
		}
		
		return output;
	}
	
	/**
	 * Returns an array of randomly selected mini-batches from the given data 
	 * set.
	 * 
	 * @param data The data set to partition into mini-batches.
	 * @param miniBatchSize The size of each mini-batch.
	 * @return An array of mini-batches from the given data set.
	 */
	public double[][][][][] getMiniBatches(double[][][][] data, 
			int miniBatchSize) {
		int numberOfMiniBatches = 
			(int) Math.ceil((double) data.length / miniBatchSize);
		double[][][][][] miniBatches = new double[numberOfMiniBatches][][][][];
		Random randomNumberGenerator = new Random();
		int[] selected = new int[data.length];
		int lastMiniBatchSize = data.length % miniBatchSize > 0 ? 
			data.length % miniBatchSize : miniBatchSize;
		
		for (int i = 0; i < numberOfMiniBatches; i++) {
			int batchSize = (i < numberOfMiniBatches - 1 ? miniBatchSize :
				lastMiniBatchSize);
			double[][][][] miniBatch = new double[batchSize][][][];
			
			for (int j = 0; j < batchSize; j++) {
				int randomDataIndex = 0;
				
				do {
					randomDataIndex = randomNumberGenerator
							.nextInt(data.length);
				} while (selected[randomDataIndex] == 1);
				
				selected[randomDataIndex] = 1;
				miniBatch[j] = data[randomDataIndex];
			}
			
			miniBatches[i] = miniBatch;
		}
		
		return miniBatches;
	}
	
	/**
	 * Performs stochastic gradient descent using the provided mini-batch and
	 * learning rate.
	 * 
	 * @param miniBatch The training data mini-batch.
	 * @param learningRate The learning rate for gradient descent.
	 * @throws Exception 
	 */
	public double[][][][] stochasticGradientDescent(double[][][][] miniBatch) 
			throws Exception {
		double weights[][][] = this.getWeights();
		double biases[][][] = this.getBiases();
		int numberOfLayers = this.getNumberOfLayers();
		
		double[][][][] gradientSum =
			new double[COST_GRADIENT_COMPONENTS][][][];
		
		for (int i = 0; i < miniBatch.length; i++) {
			// Get input to the neural network
			double[][][] data = miniBatch[i];
			double[][] input = data[0];
			double[][] label = data[1];
			
			// Save layer inputs and activations
			double[][][] layerInputs = new double[numberOfLayers - 1][][];
			double[][][] activations = new double[numberOfLayers][][];
			
			activations[0] = input;
			
			// Feedforward
			for (int l = 0; l < numberOfLayers - 1; l++) {
				double[][] previousActivation = activations[l];
				double[][] weightMatrix = weights[l];
				double[][] biasVector = biases[l];
				double[][] layerInput = Matrix.add(
					Matrix.product(weightMatrix, previousActivation),
					biasVector);
				double[][] layerActivation = Matrix.activation(layerInput);
				
				layerInputs[l] = layerInput;
				activations[l + 1] = layerActivation;
			}
			
			// Calculate gradient
			double[][][][] gradient = backpropogate(layerInputs, activations, 
				label);
			double[][][] gradientWithRespectToWeights = gradient[0];
			double[][][] gradientWithRespectToBiases = gradient[1];
			
			// Contribute to gradient sum
			if (i == 0) {
				gradientSum[0] = gradientWithRespectToWeights;
				gradientSum[1] = gradientWithRespectToBiases;
			}
			else {
				for (int j = 0; j < gradientSum[0].length; j++) {
					gradientSum[0][j] = Matrix.add(gradientSum[0][j],
							gradientWithRespectToWeights[j]);
				}
				
				for (int j = 0; j < gradientSum[1].length; j++) {
					gradientSum[1][j] = Matrix.add(gradientSum[1][j],
							gradientWithRespectToBiases[j]);
				}
			}
		}
		
		return gradientSum;
	}
	
	/**
	 * Returns the gradient of the cost function calculated through 
	 * backpropogation.
	 * 
	 * @param layerInputs An array of inputs to each layer after the first.
	 * @param activations An array of activations to each layer (including the
	 * input to the neural network.)
	 * @param label The label associated with the input to the neural network.
	 * @return The gradient.
	 * @throws Exception 
	 */
	double[][][][] backpropogate(double[][][] layerInputs, 
		double[][][] activations, double[][] label) throws Exception {
		double weights[][][] = this.getWeights();
		int biasesLength = this.getBiases().length;
		
		int layerFromEnd = 1;
		
		double[][][][] gradient = new double[COST_GRADIENT_COMPONENTS][][][];
		double[][][] gradientWithRespectToWeights = 
			new double[weights.length][][];
		double[][][] gradientWithRespectToBiases = 
			new double[biasesLength][][];
		
		double[][] currentLayerActivation = 
			activations[activations.length - layerFromEnd];
		double[][] previousLayerActivation = 
			activations[activations.length - layerFromEnd - 1];
		double[][] currentLayerInput = 
			layerInputs[layerInputs.length - layerFromEnd];
		
		double[][] costGradientWithRespectToActivation =
			Matrix.costDerivativeWithRespectToActivation(
				currentLayerActivation, label);
		double[][] activationPrime = Matrix.activationPrime(
			currentLayerInput);
		double[][] layerError = Matrix.hadamardProduct(
			costGradientWithRespectToActivation, activationPrime);
		
		double[][] layerGradientWithRespectToWeights =
			Matrix.product(layerError, 
				Matrix.transpose(previousLayerActivation));
		double[][] layerGradientWithRespectToBiases = 
			layerError;
		
		gradientWithRespectToWeights[weights.length - layerFromEnd] = 
				layerGradientWithRespectToWeights;
		gradientWithRespectToBiases[biasesLength - layerFromEnd] = 
				layerGradientWithRespectToBiases;
		
		layerFromEnd++;
		
		for (; layerFromEnd <= layerInputs.length; layerFromEnd++) {
			currentLayerActivation = 
				activations[activations.length - layerFromEnd];
			previousLayerActivation = 
				activations[activations.length - layerFromEnd - 1];
			currentLayerInput = 
				layerInputs[layerInputs.length - layerFromEnd];
			
			double[][] nextLayerWeights = 
				weights[weights.length - layerFromEnd + 1];
			activationPrime = Matrix.activationPrime(currentLayerInput);
			layerError = Matrix.hadamardProduct(
				Matrix.product(
					Matrix.transpose(nextLayerWeights), 
					layerError
				),
				activationPrime
			);
			
			layerGradientWithRespectToWeights =
				Matrix.product(layerError, 
					Matrix.transpose(previousLayerActivation));
			layerGradientWithRespectToBiases = 
				layerError;
			
			gradientWithRespectToWeights[weights.length - layerFromEnd] = 
				layerGradientWithRespectToWeights;
			gradientWithRespectToBiases[biasesLength - layerFromEnd] = 
				layerGradientWithRespectToBiases;
		}
		
		gradient[0] = gradientWithRespectToWeights;
		gradient[1] = gradientWithRespectToBiases;
		
		return gradient;
	}
	
	/**
	 * Return the derivative of the cost function with respect to the  
	 * activation of a specific neuron in the final layer. The cost function
	 * is assumed to be cross-entropy.
	 * 
	 * @param activation Activation for the final layer neuron.
	 * @param label The actual output.
	 * @return The derivative.
	 */
	public static double costDerivativeWithRespectToActivation(
			double activation, double label) {
		return (activation - label) / (activation * (1 - activation));
	}
	
	/**
	 * Returns an array of the current weight matrices in this neural network.
	 * 
	 * @return An array of the current weight matrices in this neural network.
	 */
	public double[][][] getWeights() {
		return this.weights;
	}
	
	/**
	 * Updates this neural network's array of weights.
	 * 
	 * @param weights The new array of weights for this neural network.
	 */
	public void setWeights(double[][][] weights) {
		this.weights = weights;
	}
	
	/**
	 * View the weight matrices of this neural network in a human-readable 
	 * format.
	 */
	public void printWeights() {
		double weights[][][] = this.getWeights();
		
		for (int l = 0; l < this.numberOfLayers - 1; l++) {
			double[][] weightMatrix = weights[l];
			
			System.out.println("Layer " + (l + 1) + " -> Layer " + (l + 2));
			
			Matrix.printMatrix(weightMatrix);
			
			System.out.println();
		}
	}
	
	/**
	 * Returns an array of the current bias vectors in this neural network.
	 * 
	 * @return An array of the current bias vectors in this neural network.
	 */
	public double[][][] getBiases() {
		return this.biases;
	}
	
	/**
	 * Updates this neural network's array of biases.
	 * 
	 * @param biases The new array of biases for this neural network.
	 */
	public void setBiases(double[][][] biases) {
		this.biases = biases;
	}
	
	/**
	 * View the bias vectors of this neural network in a human-readable 
	 * format.
	 */
	public void printBiases() {
		double biases[][][] = this.getBiases();
		
		for (int l = 1; l < this.numberOfLayers; l++) {
			double[][] biasVector = biases[l - 1];
			
			System.out.println("Layer " + l);
			
			Matrix.printMatrix(biasVector);
			
			System.out.println();
		}
	}
	
	/**
	 * Returns the number of layers in this neural network.
	 * 
	 * @return The number of layers in this neural network.
	 */
	public int getNumberOfLayers() {
		return this.numberOfLayers;
	}
	
	/**
	 * Returns the array representing this neural network's layers.
	 * 
	 * @return The array representing this neural network's layers.
	 */
	public int[] getLayers() {
		return this.layers;
	}
	
	/**
	 * Creates random weight matrices for each layer. For each 2D array, 
	 * the jth element in the ith array represents the weight for the jth 
	 * output in the current layer "feeding" into the ith neuron in the next 
	 * layer.
	 * 
	 * @param layers The layers of the neural network.
	 * @return An array of random weight matrices.
	 */
	public double[][][] randomWeights(int[] layers) {
		double[][][] weights = new double[layers.length - 1][][];
		Random randomNumberGenerator = new Random();
		
		for (int l = 0; l < layers.length - 1; l++) {
			int currentLayerNodes = layers[l];
			int nextLayerNodes = layers[l + 1];
			double[][] weightMatrix = new double[nextLayerNodes][];
			
			for (int n = 0; n < nextLayerNodes; n++) {
				double row[] = new double[currentLayerNodes];
				
				for (int c = 0; c < currentLayerNodes; c++) {
					row[c] = randomNumberGenerator.nextGaussian();
				}
				
				weightMatrix[n] = row;
			}
			
			weights[l] = weightMatrix;
		}
		
		return weights;
	}
	
	/**
	 * Creates a random bias vector for each layer. For each 1D array, 
	 * the ith element represents the bias for the ith neuron in the 
	 * current layer.
	 * 
	 * @param layers The layers of the neural network.
	 * @return An array of random bias vectors for each layer.
	 */
	public double[][][] randomBiases(int[] layers) {
		double[][][] biases = new double[layers.length - 1][][];
		Random randomNumberGenerator = new Random();
		
		for (int l = 1; l < layers.length; l++) {
			int currentLayerNodes = layers[l];
			double[][] column = new double[currentLayerNodes][];
			
			for (int c = 0; c < currentLayerNodes; c++) {
				column[c] = new double[] {
					randomNumberGenerator.nextGaussian()
				};
			}
			
			biases[l - 1] = column;
		}
		
		return biases;
	}
	
	/**
	 * Returns the activation function applied to a single value. This function
	 * is implemented using the sigmoid function.
	 * 
	 * @param z The input to the activation function.
	 * @return The activation function applied to a single value.
	 */
	public static double activation(double z) {
		return 1 / (1 + Math.exp(-z));
	}
	
	/**
	 * Returns the derivative of the activation funcion applied to a single 
	 * value. Assumes the activation function is a sigmoid function.
	 * 
	 * @param z The input to the derivative.
	 * @return The derivative of the activation function at the specified value.
	 */
	public static double activationPrime(double z) {
		double activationZ = activation(z);
		
		return activationZ * (1 - activationZ);
	}
}
