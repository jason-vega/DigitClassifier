/**
 * A class to train a neural network using MNIST data.
 * 
 * @author Jason Vega
 *
 */
public class Train {
	public static final int INPUT_LAYER_SIZE = 784;
	public static final int FIRST_HIDDEN_LAYER_SIZE = 70;
	public static final int SECOND_HIDDEN_LAYER_SIZE = 35;
	public static final int OUTPUT_LAYER_SIZE = 10;
	
	public static final String TRAINING_IMAGE_FILE_PATH = "";
	public static final String TRAINING_LABEL_FILE_PATH = "";
	public static final String TEST_IMAGE_FILE_PATH = "";
	public static final String TEST_IMAGE_LABEL_PATH = "";
	
	public static final int IMAGE_FILE_OFFSET = 16;
	public static final int LABEL_FILE_OFFSET = 8;
	
	public static final int MAX_TRAINING_INPUTS = 60000;
	public static final int MAX_TEST_INPUTS = 10000;
	
	public static final boolean LOAD_LABEL_VERBOSE = false;
	public static final boolean TRAIN_VERBOSE = false;
	
	public static final int INPUT_DATA_COMPONENTS = 2;
	
	public static final int MINI_BATCH_SIZE = 256;
	public static final double LEARNING_RATE = 0.1667;
	public static final int EPOCHS = 200;
	
	/**
	 * Initializes and trains a neural network using data from the MNIST data 
	 * set.
	 * 
	 * @param args Command line arguments (not used).
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		NeuralNetwork n = new NeuralNetwork(new int[]{
				INPUT_LAYER_SIZE,
				FIRST_HIDDEN_LAYER_SIZE,
				SECOND_HIDDEN_LAYER_SIZE,
				OUTPUT_LAYER_SIZE
		});
		
		LoadData trainImageLoad = new LoadData(TRAINING_IMAGE_FILE_PATH, 
				IMAGE_FILE_OFFSET, INPUT_LAYER_SIZE, MAX_TRAINING_INPUTS);
		LoadData trainLabelLoad = new LoadData(TRAINING_LABEL_FILE_PATH, 
				LABEL_FILE_OFFSET, 1, MAX_TRAINING_INPUTS, LOAD_LABEL_VERBOSE);
		LoadData testImageLoad = new LoadData(TEST_IMAGE_FILE_PATH, 
				IMAGE_FILE_OFFSET, INPUT_LAYER_SIZE, MAX_TEST_INPUTS);
		LoadData testLabelLoad = new LoadData(TEST_IMAGE_LABEL_PATH, 
				LABEL_FILE_OFFSET, 1, MAX_TEST_INPUTS, LOAD_LABEL_VERBOSE);
		
		double[][][] trainImages = trainImageLoad.getData();
		double[][][] trainLabels = trainLabelLoad.getData();
		double[][][][] trainData = new double[trainImages.length][][][];
		
		double[][][] testImages = testImageLoad.getData();
		double[][][] testLabels = testLabelLoad.getData();
		double[][][][] testData = new double[testImages.length][][][];
		
		// Zip training images and corresponding labels
		for(int i = 0; i < trainImages.length; i++) {
			double[][] trainImage = trainImages[i];
			double[][] trainLabel =
					Matrix.vectorize(1, (int) trainLabels[i][0][0], 
							OUTPUT_LAYER_SIZE);
			double[][][] data = new double[INPUT_DATA_COMPONENTS][][];
			data[0] = trainImage;
			data[1] = trainLabel;
			
			trainData[i] = data;
		}
		
		// Zip test images and corresponding labels
		for(int i = 0; i < testImages.length; i++) {
			double[][] testImage = testImages[i];
			double[][] testLabel = 
					Matrix.vectorize(1, (int) testLabels[i][0][0], 
							OUTPUT_LAYER_SIZE);
			double[][][] data = new double[INPUT_DATA_COMPONENTS][][];
			data[0] = testImage;
			data[1] = testLabel;
			
			testData[i] = data;
		}
		
		n.train(trainData, MINI_BATCH_SIZE, LEARNING_RATE, EPOCHS, testData, 
				TRAIN_VERBOSE);
	}
}
