/**
 * A helper class of matrix operations for the NeuralNetwork class.
 * 
 * @author Jason Vega
 *
 */
public class Matrix {
	public static final String UNEQUAL_DIMENSIONS_ERROR = "The dimensions of "
			+ "the matrices/vectors must be equal.";
	public static final String UNEQUAL_COLUMN_NUMBER_ERROR = "The number of " 
			+ "columns in each row is not constant for all matrices.";
	public static final String MATRIX_PRODUCT_ERROR = "The number of rows "
			+ "in the second matrix must equal the number of columns in the "
			+ "first matrix.";
	
	/**
	 * Performs matrix addition of A + B.
	 * 
	 * @param matrixA The first matrix, A.
	 * @param matrixB The second matrix, B.
	 * @return The sum of A and B.
	 * @throws Exception 
	 */
	public static double[][] add(double[][] matrixA, double[][] matrixB) 
			throws Exception {
		if (!hasConstantColumnSize(matrixA, matrixB)) {
			throw new Exception(UNEQUAL_COLUMN_NUMBER_ERROR);
		}
		
		if (!hasEqualDimensions(matrixA, matrixB)) {
			throw new Exception(UNEQUAL_DIMENSIONS_ERROR);
		}
		
		double[][] sum = new double[matrixA.length][];
		
		for (int i = 0; i < matrixA.length; i++) {
			int rowLength = matrixA[i].length;
			double row[] = new double[rowLength];
			
			for (int j = 0; j < rowLength; j++) {
				row[j] = matrixA[i][j] + matrixB[i][j];
			}
			
			sum[i] = row;
		}
		
		return sum;
	}
	
	/**
	 * Performs matrix subtraction A - B.
	 * 
	 * @param matrixA The first matrix, A.
	 * @param matrixB The second matrix, B.
	 * @return The difference between A and B.
	 * @throws Exception
	 */
	public static double[][] subtract(double[][] matrixA, double[][] matrixB) 
			throws Exception {
		if (!hasConstantColumnSize(matrixA, matrixB)) {
			throw new Exception(UNEQUAL_COLUMN_NUMBER_ERROR);
		}
		
		if (!hasEqualDimensions(matrixA, matrixB)) {
			throw new Exception(UNEQUAL_DIMENSIONS_ERROR);
		}
		
		double[][] difference = new double[matrixA.length][];
		
		for (int i = 0; i < matrixA.length; i++) {
			int rowLength = matrixA[i].length;
			double row[] = new double[rowLength];
			
			for (int j = 0; j < rowLength; j++) {
				row[j] = matrixA[i][j] - matrixB[i][j];
			}
			
			difference[i] = row;
		}
		
		return difference;
	}
	
	/**
	 * Performs matrix multiplication of A * B. If A is a row vector and B a 
	 * column vector, then the inner product is computed. If A is a column
	 * vector and B a row vector, then the outer product is computed.
	 * 
	 * @param matrixA The first matrix, A.
	 * @param matrixB The second matrix, B.
	 * @return The product of A and B.
	 * @throws Exception 
	 */
	public static double[][] product(double[][] matrixA, double[][] matrixB) 
			throws Exception {
		if (!hasConstantColumnSize(matrixA, matrixB)) {
			throw new Exception(UNEQUAL_COLUMN_NUMBER_ERROR);
		}
		
		if (matrixB.length != matrixA[0].length) {
			throw new Exception(MATRIX_PRODUCT_ERROR);
		}
		
		int productRowSize = matrixA.length;
		int productColumnSize = matrixB[0].length;
		double[][] product = new double[productRowSize][];
		
		for (int r = 0; r < productRowSize; r++) {
			double[] row = new double[productColumnSize];
			
			for (int c = 0; c < productColumnSize; c++) {
				for (int i = 0; i < matrixB.length; i++) {
					row[c] += matrixA[r][i] * matrixB[i][c];
				}
			}
			
			product[r] = row;
		}
		
		return product;
	}
	
	/**
	 * Returns the Hadamard product of two matrices.
	 * 
	 * @param vectorA The first matrix.
	 * @param vectorB The second matrix.
	 * @return The Hadamard product.
	 * @throws Exception 
	 */
	public static double[][] hadamardProduct(double[][] matrixA, double[][] matrixB) 
			throws Exception {
		if (!hasConstantColumnSize(matrixA, matrixB)) {
			throw new Exception(UNEQUAL_COLUMN_NUMBER_ERROR);
		}
		
		if (!hasEqualDimensions(matrixA, matrixB)) {
			throw new Exception(UNEQUAL_DIMENSIONS_ERROR);
		}
		
		double[][] result = new double[matrixA.length][];
		
		for (int i = 0; i < matrixA.length; i++) {
			double[] row = new double[matrixA[i].length];
			
			for (int j = 0; j < matrixA[i].length; j++) {
				row[j] = matrixA[i][j] * matrixB[i][j];
			}
			
			result[i] = row;
		}
		
		return result;
	}
	
	/**
	 *  Returns the transpose of the given matrix.
	 *
	 * @param matrix The matrix to transpose.
	 * @return The transposed matrix.
	 * @throws Exception 
	 */
	public static double[][] transpose(double[][] matrix) throws Exception {
		if (!hasConstantColumnSize(matrix)) {
			throw new Exception(UNEQUAL_COLUMN_NUMBER_ERROR);
		}
		
		int transposeRowSize = matrix[0].length;
		int transposeColumnSize = matrix.length;
		double[][] result = new double[transposeRowSize][];
		
		for (int i = 0; i < transposeRowSize; i++) {
			double[] row = new double[transposeColumnSize];
			
			for (int j = 0; j < transposeColumnSize; j++) {
				row[j] = matrix[j][i];
			}
			
			result[i] = row;
		}
		
		return result;
	}
	
	/**
	 * Returns a new matrix scaled by the given factor.
	 * 
	 * @param factor The factor to scale by.
	 * @param matrix The matrix to scale.
	 * @return The new scaled scaled matrix.
	 */
	public static double[][] scale(double factor, double[][] matrix) {
		double[][] result = new double[matrix.length][];
		
		for (int i = 0; i < matrix.length; i++) {
			double[] row = new double[matrix[i].length];
			
			for (int j = 0; j < matrix[i].length; j++) {
				row[j] = matrix[i][j] * factor;
			}
			
			result[i] = row;
		}
		
		return result;
	}
	
	/**
	 * Returns the index of the greatest element in the vector.
	 * 
	 * @param vector A given column vector.
	 * @return The index.
	 */
	public static int argmax(double[][] vector) {
		int maxIndex = 0;
		double max = vector[0][0];
		
		for (int i = 1; i < vector.length; i++) {
			double value = vector[i][0];
			
			if (value > max) {
				maxIndex = i;
				max = value;
			}
		}
		
		return maxIndex;
	}
	
	/**
	 * Returns a new matrix with the activation function applied to every 
	 * value in the given 2D matrix.
	 * 
	 * @param matrix The matrix to pass into the activation function.
	 * @return The resulting matrix.
	 */
	public static double[][] activation(double[][] matrix) {
		double[][] result = new double[matrix.length][];
		
		for (int i = 0; i < matrix.length; i++) {
			double[] row = new double[matrix[i].length];
			
			for (int j = 0; j < matrix[i].length; j++) {
				row[j] = NeuralNetwork.activation(matrix[i][j]);
			}
			
			result[i] = row;
		}
		
		return result;
	}
	
	/**
	 * Returns a new matrix with the activation prime function applied to every 
	 * value in the given 2D matrix.
	 * 
	 * @param matrix The matrix to pass into the activation prime function.
	 * @return The resulting matrix.
	 */
	public static double[][] activationPrime(double[][] matrix) {
		double[][] result = new double[matrix.length][];
		
		for (int i = 0; i < matrix.length; i++) {
			double[] row = new double[matrix[i].length];
			
			for (int j = 0; j < matrix[i].length; j++) {
				row[j] = NeuralNetwork.activationPrime(matrix[i][j]);
			}
			
			result[i] = row;
		}
		
		return result;
	}
	
	/**
	 * Returns a new matrix with the cost derivative function applied to all
	 * elements of the given activation and label matrices.
	 * 
	 * @param activationMatrix The activation matrix.
	 * @param labelMatrix The label (output in the final layer) matrix.
	 * @return The new matrix.
	 * @throws Exception 
	 */
	public static double[][] costDerivativeWithRespectToActivation(
		double[][] activationMatrix, double[][] labelMatrix) 
			throws Exception {
		if (!hasConstantColumnSize(activationMatrix, labelMatrix)) {
			throw new Exception(UNEQUAL_COLUMN_NUMBER_ERROR);
		}
		
		if (!hasEqualDimensions(activationMatrix, labelMatrix)) {
			throw new Exception(UNEQUAL_DIMENSIONS_ERROR);
		}
		
		double[][] gradient = new double[activationMatrix.length][];
		
		for (int i = 0; i < activationMatrix.length; i++) {
			double[] row = new double[activationMatrix[i].length];
			
			for (int j = 0; j < activationMatrix[i].length; j++) {
				row[j] = NeuralNetwork.costDerivativeWithRespectToActivation(
					activationMatrix[i][j], labelMatrix[i][j]);
			}
			
			gradient[i] = row;
		}
		
		return gradient;
	}
	
	/**
	 * Vectorizes the given value. All other values in the vector except the 
	 * one given at the specified index are 0.
	 * 
	 * @param value The value to vectorize.
	 * @param index The row to place the value in.
	 * @return The vector.
	 */
	public static double[][] vectorize(double value, int index, 
			int rows) {
		double[][] vector = zeros(rows, 1);
		
		vector[index][0] = value;
		
		return vector;
	}
	
	/**
	 * Creates a matrix of zeros with specified dimensions.
	 * 
	 * @param rows The number of rows in the matrix.
	 * @param columns The number of columns in the matrix.
	 * @return The matrix.
	 */
	public static double[][] zeros(int rows, int columns) {
		double[][] result = new double[rows][];
		
		for (int i = 0; i < rows; i++) {
			double[] row = new double[columns];
			
			for (int j = 0; j < columns; j++) {
				row[j] = 0;
			}
			
			result[i] = row;
		}
		
		return result;
	}
	
	/**
	 * Verifies whether the dimensions of the two given matrices are equal.
	 * 
	 * @param matrixA The first matrix.
	 * @param matrixB The second matrix.
	 * @return Whether the dimensions of the two given matrices are equal.
	 */
	public static boolean hasEqualDimensions(double[][] matrixA, 
			double[][] matrixB) {
		boolean equal = true;
		
		if (matrixA.length != matrixB.length) {
			equal = false;
		}
		
		for (int i = 0; equal && i < matrixA.length; i++) {
			if (matrixA[i].length != matrixB[i].length) {
				equal = false;
			}
		}
		
		return equal;
	}
	
	/**
	 * Verifies whether the number of columns for each row in each given matrix
	 * is constant.
	 * 
	 * @param matrix
	 * @return Whether the number of columns for each row in each given matrix
	 * is constant.
	 */
	public static boolean hasConstantColumnSize(double[][]... matrices) {
		boolean constant = true;
		int columns = 0;
		
		for (int i = 0; constant && i < matrices.length; i++) {
			double[][] matrix = matrices[i];
			
			for (int j = 0; constant && j < matrix.length; j++) {
				int columnSize = matrix[j].length;
				
				if (j > 0 && columnSize != columns) {
					constant = false;
				}
				
				columns = columnSize;
			}
		}
		
		return constant;
	}
	
	/**
	 * Prints the given matrix in human-readable form. If the matrix is a 
	 * column vector, then the vector is printed.
	 * 
	 * @param matrix The matrix to print.
	 */
	public static void printMatrix(double[][] matrix) {
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[i].length; j++) {
				System.out.print(matrix[i][j] + "\t");
			}
			
			System.out.println();
		}
	}
}
