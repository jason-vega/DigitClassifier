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
	public static final long NANOSECOND_RATIO = 1000000000;
	public static final String EPOCH_MESSAGE = "EPOCH";
	public static final String ACCURACY_MESSAGE = "Accuracy:";
	public static final String TIME_ELAPSED_MESSAGE = "Time elapsed:";
	public static final String MINI_BATCH_UNIT = "mini-batch";
	public static final String TEST_IMAGE_UNIT = "test image";
	public static final String TRAIN_IMAGE_UNIT = "training image";
	public static final String TRAINING_PROGRESS_MESSAGE = 
		"Training progress:";
	public static final String TESTING_PROGRESS_MESSAGE = 
		"Testing progress:";
	public static final String TRAINING_START_MESSAGE = 
			"s (since training start), ";
	public static final String EPOCH_START_MESSAGE = 
			"s (since epoch start)";
	public static final String TRAIN_ACCURACY_MESSAGE = 
			"% (training)";
	public static final String TEST_ACCURACY_MESSAGE = 
			"% (test), ";
	
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
	 * @param trainVerbose Whether or not to output training processing 
         * data.
	 * @param testVerbose Whether or not to output test processing data.
	 * @throws Exception 
	 */
	public void train(double[][][][] trainingData, int miniBatchSize, 
			double learningRate, int epochs, double[][][][] testData, 
			boolean trainVerbose, boolean testVerbose) 
					throws Exception {
		long initialTimeStart = System.nanoTime();
		
		for (int i = 1; i <= epochs; i++) {
			double[][][][][] miniBatches = this.getMiniBatches(trainingData, 
					miniBatchSize);
			long initialTimeEpoch = System.nanoTime();
			String currentProgressBar = "";
			
			System.out.println("-- " + EPOCH_MESSAGE + " " + i + "/" + epochs + 
				" --");
			
			for (int j = 0; j < miniBatches.length; j++) {
				double[][][][] miniBatch = miniBatches[j];
				
				this.stochasticGradientDescent(miniBatch, learningRate);
				
				if (trainVerbose) {
					if (j > 0) {
						LoadData.deleteProgressBar(currentProgressBar);
					}
					
					currentProgressBar = 
						LoadData.getProgressBar(TRAINING_PROGRESS_MESSAGE, j + 1, 
								miniBatches.length, MINI_BATCH_UNIT);
					
					System.out.print(currentProgressBar);
					
					if (j == miniBatches.length - 1) {
						System.out.print('\n');
					}
				}
			}
			
			long elapsedTimeStart = System.nanoTime() - initialTimeStart;
			long elapsedTimeEpoch = System.nanoTime() - initialTimeEpoch;
			
			System.out.println(ACCURACY_MESSAGE + " " + 
					this.getAccuracy(testData, TEST_IMAGE_UNIT, testVerbose) + 
					TEST_ACCURACY_MESSAGE + 
					this.getAccuracy(trainingData, TRAIN_IMAGE_UNIT, testVerbose) + 
					TRAIN_ACCURACY_MESSAGE);
			System.out.println(TIME_ELAPSED_MESSAGE + " " + 
				elapsedTimeStart / NANOSECOND_RATIO + TRAINING_START_MESSAGE +
				elapsedTimeEpoch / NANOSECOND_RATIO + EPOCH_START_MESSAGE +
				'\n');
		}
	}
	
	/**
	 * Calculates and returns the accuracy (%) of this neural network.
	 * 
	 * @param dataSet The data to test for accuracy.
	 * @param unit The unit of data for the data set.
	 * @param verboes Whether or not to test in verbose mode.
	 * @return The accuracy rate as a percentage.
	 * @throws Exception 
	 */
	public double getAccuracy(double[][][][] dataSet, String unit, 
			boolean verbose) throws Exception {
		int correct = 0;
		String currentProgressBar = "";
		
		for (int i = 0; i < dataSet.length; i++) {
			double data[][][] = dataSet[i];
			double[][] input = data[0];
			double[][] label = data[1];
			double[][] output = forwardPass(input);
			int resultIndex = Matrix.argmax(output);
			
			if (label[resultIndex][0] == 1) {
				correct++;
			}
			
			if (verbose) {
				if (i > 0) {
					LoadData.deleteProgressBar(currentProgressBar);
				}
				
				currentProgressBar = 
					LoadData.getProgressBar(TESTING_PROGRESS_MESSAGE, i + 1, 
							dataSet.length, unit);
				
				System.out.print(currentProgressBar);
				
				if (i == dataSet.length - 1) {
					System.out.print('\n');
				}
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
	public void stochasticGradientDescent(double[][][][] miniBatch, 
			double learningRate) throws Exception {
		double weights[][][] = this.getWeights();
		double biases[][][] = this.getBiases();
		int numberOfLayers = this.getNumberOfLayers();
		
		double[][][] layerInputMatrices = new double[numberOfLayers - 1][][];
		double[][][] activationMatrices = new double[numberOfLayers][][];
		
		double[][][] inputActivations = new double[miniBatch.length][][];
		double[][][] labels = new double[miniBatch.length][][];
		
		// Extract input activations and labels from mini-batch
		for (int i = 0; i < miniBatch.length; i++) {
			inputActivations[i] = miniBatch[i][0];
			labels[i] = miniBatch[i][1];
		}
		
		activationMatrices[0] = Matrix.concatenate(inputActivations);
		
		double[][] labelMatrix = 
				Matrix.concatenate(labels);
		
		// Forward pass
		for (int l = 1; l < numberOfLayers; l++) {
			layerInputMatrices[l - 1] = 
					Matrix.add(
						Matrix.product(weights[l - 1], 
								activationMatrices[l - 1]),
						Matrix.tile(biases[l - 1], 
								miniBatch.length)
					);
			activationMatrices[l] = 
					Matrix.activation(layerInputMatrices[l - 1]);
		}
		
		// Calculate the gradient sum
		double[][][][] gradientSum = backpropogate(layerInputMatrices, 
				activationMatrices, labelMatrix);
		double[][][] gradientSumWithRespectToWeights = gradientSum[0];
		double[][][] gradientSumWithRespectToBiases = gradientSum[1];
		
		// Perform gradient descent
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
	
	/**
	 * Returns the sum of cost function gradients across all mini-batch inputs 
	 * calculated through backpropogation.
	 * 
	 * @param layerInputMatrices An array of input matrices to each layer after
	 * the first.
	 * @param activationMatrices An array of activation matrices to each layer 
	 * (including the input to the neural network.)
	 * @param labelMatrix The label matrix associated with the input activation 
	 * matrix to the neural network.
	 * @return The gradient sum.
	 * @throws Exception 
	 */
	double[][][][] backpropogate(double[][][] layerInputMatrices, 
			double[][][] activationMatrices, double[][] labelMatrix) 
					throws Exception {
		double weights[][][] = this.getWeights();
		int biasesLength = this.getBiases().length;
		
		int layerFromEnd = 1;
		
		double[][][][] gradientSum = 
			new double[COST_GRADIENT_COMPONENTS][][][];
		double[][][] gradientSumWithRespectToWeights = 
			new double[weights.length][][];
		double[][][] gradientSumWithRespectToBiases = 
			new double[biasesLength][][];
		
		double[][] currentLayerActivationMatrix = 
			activationMatrices[activationMatrices.length - layerFromEnd];
		double[][] previousLayerActivationMatrix = 
			activationMatrices[activationMatrices.length - layerFromEnd - 1];
		double[][] currentLayerInputMatrix = 
			layerInputMatrices[layerInputMatrices.length - layerFromEnd];
		
		double[][] costGradientWithRespectToActivation =
			Matrix.costDerivativeWithRespectToActivation(
					currentLayerActivationMatrix, labelMatrix);
		double[][] activationPrime = Matrix.activationPrime(
				currentLayerInputMatrix);
		double[][] layerError = Matrix.hadamardProduct(
				costGradientWithRespectToActivation, activationPrime);
		
		gradientSumWithRespectToWeights[weights.length - layerFromEnd] = 
				Matrix.product(layerError, 
						Matrix.transpose(previousLayerActivationMatrix));
		gradientSumWithRespectToBiases[biasesLength - layerFromEnd] = 
				Matrix.sumColumns(layerError);
		
		layerFromEnd++;
		
		for (; layerFromEnd <= layerInputMatrices.length; layerFromEnd++) {
			previousLayerActivationMatrix = 
				activationMatrices[activationMatrices.length - layerFromEnd - 
				                   1];
			currentLayerInputMatrix = 
				layerInputMatrices[layerInputMatrices.length - layerFromEnd];
			
			double[][] nextLayerWeights = 
				weights[weights.length - layerFromEnd + 1];
			activationPrime = Matrix.activationPrime(currentLayerInputMatrix);
			layerError = Matrix.hadamardProduct(
				Matrix.product(
					Matrix.transpose(nextLayerWeights), 
					layerError
				),
				activationPrime
			);
			
			gradientSumWithRespectToWeights[weights.length - layerFromEnd] = 
					Matrix.product(layerError, 
							Matrix.transpose(previousLayerActivationMatrix));
			gradientSumWithRespectToBiases[biasesLength - layerFromEnd] = 
					Matrix.sumColumns(layerError);
		}
		
		gradientSum[0] = gradientSumWithRespectToWeights;
		gradientSum[1] = gradientSumWithRespectToBiases;
		
		return gradientSum;
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
