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
	public static final String ACCURACY_MESSAGE = "ACCURACY:";
	public static final String TIME_ELAPSED_MESSAGE = "TIME ELAPSED:";
	public static final String MINI_BATCH_UNIT = "mini-batch";
	public static final String TEST_IMAGE_UNIT = "test image";
	public static final String TRAIN_IMAGE_UNIT = "training image";
	public static final String TRAINING_MESSAGE = "Training...";
	public static final String TESTING_MESSAGE = "Testing...";
	public static final String TRAINING_PROGRESS_MESSAGE = 
		"Training progress:";
	public static final String TESTING_PROGRESS_MESSAGE = 
		"Testing progress:";
	public static final String TRAIN_START_MESSAGE = 
			"s (since training start), ";
	public static final String EPOCH_START_MESSAGE = 
			"s (since epoch start)";
	public static final String ACCURACY_UNIT = "%";
	public static final String TEST_ACCURACY_TITLE = "(test)";
	public static final String TRAIN_ACCURACY_TITLE = "(training)";
	
	private int[] layers;
	private double[][][] weights;
	private double[][][] biases;
	private int numberOfLayers;
	
	/**
	 * Constructs a new neural network representation where the number of nodes
	 * in each layer is specified using the layers array.
	 * 
	 * @param layers An array representing the number of nodes in each layer.
	 */
	public NeuralNetwork(int... layers) {
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
	 * @param miniBatchSize The (maximum) size of each mini-batch.
	 * @param learningRate The learning rate for the optimization method.
	 * @param regularization The regularization parameter for the optimization 
	 * method.
	 * @param epochs The number of training sessions.
	 * @param learningRateDecay The decay rate for the learning rate schedule.
	 * @param testData The test data, including the input and corresponding
	 * labels.
	 * @param trainVerbose Whether or not to output training processing 
         * data.
	 * @param getTestAccuracy Whether or not to get the accuracy on the test 
	 * set.
	 * @param getTrainingAccuracy Whether or not to get the accuracy on the  
	 * training set.
	 * @param testVerbose Whether or not to output test processing data.
	 * @throws Exception 
	 */
	public void train(double[][][][] trainingData, int miniBatchSize, 
			double learningRate, double regularization, int epochs,
			double learningRateDecay, double[][][][] testData, 
			boolean trainVerbose, boolean getTestAccuracy,
			boolean getTrainingAccuracy, boolean testVerbose) 
					throws Exception {
		long initialTimeStart = System.nanoTime();
		
		for (int i = 1; i <= epochs; i++) {
			double[][][][][] miniBatches = this.getMiniBatches(trainingData, 
					miniBatchSize);
			long initialTimeEpoch = System.nanoTime();
			String currentProgressBar = "";
			
			System.out.println(EPOCH_MESSAGE + " " + i + "/" + epochs);
			System.out.println("\n" + Train.SEPARATOR + "\n");
			System.out.println(TRAINING_MESSAGE);
			
			for (int j = 0; j < miniBatches.length; j++) {
				double[][][][] miniBatch = miniBatches[j];
				
				this.stochasticGradientDescent(miniBatch, 
					this.learningSchedule(learningRate, (i - 1), 
						learningRateDecay), regularization, 
					trainingData[0].length);
				
				// Display progress bar
				if (trainVerbose) {
					currentProgressBar = 
						LoadData.updateProgressBar(currentProgressBar, 
								LoadData.PROGRESS_MESSAGE, (j + 1), 
								miniBatches.length, MINI_BATCH_UNIT);
					
					System.out.print(currentProgressBar);
				}
			}
			
			System.out.println(LoadData.DONE_MESSAGE + "\n");
			
			if (getTestAccuracy || getTrainingAccuracy) {
				System.out.println(TESTING_MESSAGE);
			}
			
			double testAccuracy = (getTestAccuracy ? this.getAccuracy(
					testData, TEST_IMAGE_UNIT, testVerbose) : 0);
			double trainingAccuracy = (getTrainingAccuracy ? this.getAccuracy(
					trainingData, TRAIN_IMAGE_UNIT, testVerbose) : 0);
			
			if (getTestAccuracy || getTrainingAccuracy) {
				System.out.println(LoadData.DONE_MESSAGE + "\n");
				System.out.print(ACCURACY_MESSAGE + " ");
			}
			
			// Test accuracy output
			if (getTestAccuracy) {
				System.out.print(testAccuracy + ACCURACY_UNIT + " "  +
						TEST_ACCURACY_TITLE +
						(getTrainingAccuracy ? ", " : "\n"));
			}
			
			// Training accuracy output
			if (getTrainingAccuracy) {
				System.out.print(trainingAccuracy + ACCURACY_UNIT + " " + 
						TRAIN_ACCURACY_TITLE + "\n");
			}
			
			long elapsedTimeStart = System.nanoTime() - initialTimeStart;
			long elapsedTimeEpoch = System.nanoTime() - initialTimeEpoch;
			
			// Elapsed time output
			System.out.println(TIME_ELAPSED_MESSAGE + " " + 
				elapsedTimeStart / NANOSECOND_RATIO + TRAIN_START_MESSAGE +
				elapsedTimeEpoch / NANOSECOND_RATIO + EPOCH_START_MESSAGE);
			
			System.out.println("\n" + Train.SEPARATOR + "\n\n");
		}
	}
	
	/**
	 * Performs stochastic gradient descent using the provided mini-batch, 
	 * learning rate and regularization parameter.
	 * 
	 * @param miniBatch The training data mini-batch.
	 * @param learningRate The learning rate for gradient descent.
	 * @param regularization The regularization parameter for the optimization 
	 * method.
	 * @param trainingSetSize The size of the training set.
	 * 
	 * @throws Exception 
	 */
	public void stochasticGradientDescent(double[][][][] miniBatch, 
			double learningRate, double regularization, int trainingSetSize) 
					throws Exception {
		double weights[][][] = this.getWeights();
		double biases[][][] = this.getBiases();
		int numberOfLayers = this.getNumberOfLayers();
		
		double[][][] layerInputMatrices = new double[numberOfLayers - 1][][];
		double[][][] activationMatrices = new double[numberOfLayers][][];
		
		activationMatrices[0] = Matrix.concatenate(miniBatch[0]);
		
		double[][] labelMatrix = 
				Matrix.concatenate(miniBatch[1]);

		// Forward pass
		for (int l = 1; l < numberOfLayers; l++) {
			layerInputMatrices[l - 1] = 
					Matrix.add(
						Matrix.product(weights[l - 1], 
								activationMatrices[l - 1]),
						Matrix.tile(biases[l - 1], 
								miniBatch[0].length)
					);
			activationMatrices[l] = 
					Matrix.activation(layerInputMatrices[l - 1]);
		}
		
		// Calculate the gradient sum
		double[][][][] gradientSum = backpropogate(layerInputMatrices, 
				activationMatrices, labelMatrix);
		double[][][] gradientSumWithRespectToWeights = gradientSum[0];
		double[][][] gradientSumWithRespectToBiases = gradientSum[1];
		
		// Perform gradient descent with L2 regularization
		for (int k = 0; k < numberOfLayers - 1; k++) {
			weights[k] = Matrix.subtract(
				Matrix.scale(1 - learningRate * regularization / 
						trainingSetSize, weights[k]), 
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
	 * Returns a learning rate in accordance to the learning schedule.
	 * 
	 * @param learningRate The initial learning rate.
	 * @param iteration The current iteration.
	 * @param decayRate The rate of decay for the learning schedule.
	 * @return the resulting learning rate.
	 */
	public double learningSchedule(double learningRate, int iteration, 
			double decayRate) {
		return learningRate * 1 / (1 + decayRate * iteration);
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
		
		double[][][] images = dataSet[0];
		double[][][] labels = dataSet[1];
		
		for (int i = 0; i < images.length; i++) {
			double[][] output = this.forwardPass(images[i]);
			int resultIndex = Matrix.argmax(output);
			
			if (labels[i][resultIndex][0] == 1) {
				correct++;
			}
			
			if (verbose) {
				currentProgressBar = 
					LoadData.updateProgressBar(currentProgressBar,
							LoadData.PROGRESS_MESSAGE, i + 1, images.length, 
							unit);
				
				System.out.print(currentProgressBar);
			}
		}
		
		return (double) correct / images.length * PERCENTAGE_RATIO;
	}
	
	/**
	 * Perform a single forward pass through the network with the given input.
	 * 
	 * @param input The input to this neural network. Can be a single vector or
	 * a matrix representing multiple inputs.
	 * @return The output of the neural network.
	 * @throws Exception 
	 */
	public double[][] forwardPass(double[][] input) throws Exception {
		double[][][] weights = this.getWeights();
		double[][][] biases = this.getBiases();
		double output[][] = input;
		
		for (int l = 0; l < this.getNumberOfLayers() - 1; l++) {
			double[][] weightMatrix = weights[l];
			double[][] biasMatrix = Matrix.tile(biases[l], 
					input[0].length);
			
			output = Matrix.activation(
				Matrix.add(
					Matrix.product(weightMatrix, output), 
					biasMatrix
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
		double[][][] images = data[0];
		double[][][] labels = data[1];
		int numberOfMiniBatches = 
			(int) Math.ceil((double) images.length / miniBatchSize);
		double[][][][][] miniBatches = new double[numberOfMiniBatches][][][][];
		Random randomNumberGenerator = new Random();
		int[] selected = new int[images.length];
		int lastMiniBatchSize = images.length % miniBatchSize > 0 ? 
			images.length % miniBatchSize : miniBatchSize;
		
		for (int i = 0; i < numberOfMiniBatches; i++) {
			int batchSize = (i < numberOfMiniBatches - 1 ? miniBatchSize :
				lastMiniBatchSize);
			double[][][] miniBatchImages = new double[batchSize][][];
			double[][][] miniBatchLabels = new double[batchSize][][];
			double[][][][] miniBatch = new double[][][][] {
				miniBatchImages,
				miniBatchLabels
			};
			
			for (int j = 0; j < batchSize; j++) {
				int randomDataIndex = 0;
				
				do {
					randomDataIndex = randomNumberGenerator
							.nextInt(images.length);
				} while (selected[randomDataIndex] == 1);
				
				selected[randomDataIndex] = 1;
				miniBatchImages[j] = images[randomDataIndex];
				miniBatchLabels[j] = labels[randomDataIndex];
			}
			
			miniBatches[i] = miniBatch;
		}
		
		return miniBatches;
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
					row[c] = randomNumberGenerator.nextGaussian() / 
						Math.sqrt(currentLayerNodes);
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
