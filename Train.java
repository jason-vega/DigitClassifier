/**
 * A class to train a neural network using MNIST data.
 * 
 * @author Jason Vega
 *
 */
public class Train {
	public static final int INPUT_LAYER_SIZE = 784;
	public static final int FIRST_HIDDEN_LAYER_SIZE = 100;
	public static final int OUTPUT_LAYER_SIZE = 10;
	
	public static final String TRAINING_IMAGE_FILE_PATH = "C:\\Users\\Jason Vega\\Downloads\\train-images-idx3-ubyte";
	public static final String TRAINING_LABEL_FILE_PATH = "C:\\Users\\Jason Vega\\Downloads\\train-labels-idx1-ubyte";
	public static final String TEST_IMAGE_FILE_PATH = "C:\\Users\\Jason Vega\\Downloads\\t10k-images-idx3-ubyte";
	public static final String TEST_IMAGE_LABEL_PATH = "C:\\Users\\Jason Vega\\Downloads\\t10k-labels-idx1-ubyte";
	
	public static final int IMAGE_FILE_OFFSET = 16;
	public static final int LABEL_FILE_OFFSET = 8;
	
	public static final int MAX_TRAINING_INPUTS = 60000;
	public static final int MAX_TEST_INPUTS = 10000;
	
	public static final boolean LOAD_IMAGE_VERBOSE = false;
	public static final boolean LOAD_LABEL_VERBOSE = false;
	public static final boolean TRAIN_VERBOSE = false;
	public static final boolean TEST_VERBOSE = false;
	
	public static final boolean GET_TEST_ACCURACY_EACH_EPOCH = true;
	public static final boolean GET_TRAINING_ACCURACY_EACH_EPOCH = false;
	
	public static final int INPUT_DATA_COMPONENTS = 2;
	
	public static final int MINI_BATCH_SIZE = 10;
	public static final double LEARNING_RATE = 0.5;
	public static final int EPOCHS = 30;
	
	public static final boolean NORMALIZE_PIXEL_DATA = true;
	public static final boolean NORMALIZE_LABEL_DATA = false;
	
	public static final String IMAGE_UNIT = "image";
	public static final String LABEL_UNIT = "label";
	
	public static final String START_MESSAGE = "MNIST Digit Classifier by "
			+ "Jason Vega";
	public static final String SEPARATOR = 
			"------------------------------------";
	
	public static final String FINAL_TEST_ACCURACY_MESSAGE = 
			"FINAL TEST ACCURACY: ";
	
	/**
	 * Initializes and trains a neural network using data from the MNIST data 
	 * set.
	 * 
	 * @param args Command line arguments (not used).
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		System.out.println(START_MESSAGE + '\n');
		System.out.println(SEPARATOR + '\n');
		
		NeuralNetwork net = new NeuralNetwork(
				INPUT_LAYER_SIZE,
				FIRST_HIDDEN_LAYER_SIZE,
				OUTPUT_LAYER_SIZE
		);
		
		LoadData trainImageLoad = new LoadData(TRAINING_IMAGE_FILE_PATH, 
				IMAGE_FILE_OFFSET, INPUT_LAYER_SIZE, MAX_TRAINING_INPUTS, 
				IMAGE_UNIT, NORMALIZE_PIXEL_DATA, LOAD_IMAGE_VERBOSE);
		LoadData trainLabelLoad = new LoadData(TRAINING_LABEL_FILE_PATH, 
				LABEL_FILE_OFFSET, 1, MAX_TRAINING_INPUTS, LABEL_UNIT, 
				NORMALIZE_LABEL_DATA, LOAD_LABEL_VERBOSE);
		LoadData testImageLoad = new LoadData(TEST_IMAGE_FILE_PATH, 
				IMAGE_FILE_OFFSET, INPUT_LAYER_SIZE, MAX_TEST_INPUTS,
				IMAGE_UNIT, NORMALIZE_PIXEL_DATA, LOAD_IMAGE_VERBOSE);
		LoadData testLabelLoad = new LoadData(TEST_IMAGE_LABEL_PATH, 
				LABEL_FILE_OFFSET, 1, MAX_TEST_INPUTS, LABEL_UNIT,
				NORMALIZE_LABEL_DATA, LOAD_LABEL_VERBOSE);
		
		System.out.println(SEPARATOR + "\n\n");
		
		double[][][] trainImages = trainImageLoad.getData();
		double[][][] trainLabels = vectorizeLabels(trainLabelLoad.getData());
		double[][][][] trainData = new double[][][][] {
			trainImages,
			trainLabels
		};
		
		double[][][] testImages = testImageLoad.getData();
		double[][][] testLabels = vectorizeLabels(testLabelLoad.getData());
		double[][][][] testData = new double[][][][] {
			testImages,
			testLabels
		};
		
		net.train(trainData, MINI_BATCH_SIZE, LEARNING_RATE, EPOCHS, testData, 
		TRAIN_VERBOSE, GET_TEST_ACCURACY_EACH_EPOCH, 
		GET_TRAINING_ACCURACY_EACH_EPOCH, TEST_VERBOSE);
		
		if (!GET_TEST_ACCURACY_EACH_EPOCH) {
			double testAccuracy = net.getAccuracy(testData, 
					NeuralNetwork.TEST_IMAGE_UNIT, TEST_VERBOSE);
			
			System.out.println(FINAL_TEST_ACCURACY_MESSAGE + " " + 
					testAccuracy + NeuralNetwork.ACCURACY_UNIT + "\n");
		}
	}
	
	/**
	 * Restructures label data for use in the neural network.
	 * 
	 * @param labels Label data.
	 * @return the restructured label data.
	 */
	public static double[][][] vectorizeLabels(double[][][] labels) {
		double[][][] result = new double[labels.length][][];
		
		for(int i = 0; i < labels.length; i++) {
			result[i] = Matrix.vectorize(1, (int) labels[i][0][0], 
							OUTPUT_LAYER_SIZE);
		}
		
		return result;
	}
}
