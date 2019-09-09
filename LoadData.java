import java.io.File;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * A helper class to convert binary data from the MNIST data set into Java 2D
 * arrays. 
 * 
 * @author Jason Vega
 *
 */
public class LoadData {
	public static final String START_MESSAGE = "Reading from file";
	public static final String PROGRESS_MESSAGE = "Progress:";
	public static final String DONE_MESSAGE = "Done.";
	public static final char PROGRESS_BLOCK = '#';
	public static final char EMPTY_PROGRESS_BLOCK = ' ';
	public static final int MAX_LINE_WIDTH = 80;
	public static final int MAX_BYTE_VALUE = 255;
	
	private String filePath;
	private int fileOffset;
	private int blockLength;
	private int maxBlocks;
	private String unit;
	
	private double[][][] data;
	
	/**
	 * Loads binary data from file.
	 * 
	 * @param filePath The path to the data file.
	 * @param fileOffset The byte to start reading at.
	 * @param blockLength The length of each block of data.
	 * @param maxBlocks An upper limit on the number of blocks read.
	 * @param unit the The unit of data.
	 * @param normalize Whether or not to normalize the data.
	 * @param verbose Whether or not to output loading progress info.
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public LoadData(String filePath, int fileOffset, int blockLength, 
			int maxBlocks, String unit, boolean normalize, boolean verbose) 
			throws IOException, InterruptedException {
		this.filePath = filePath;
		this.fileOffset = fileOffset;
		this.blockLength = blockLength;
		this.maxBlocks = maxBlocks;
		
		this.data = getDataFromBinary(filePath, fileOffset, blockLength, 
				maxBlocks, unit, normalize, verbose);
	}
	
	/**
	 * Reads binary data from file and saves each block of data into a 2D 
	 * array.
	 * 
	 * @param filePath The path to the data file.
	 * @param fileOffset The byte to start reading at.
	 * @param blockLength The length of each block of data.
	 * @param maxBlocks An upper limit on the number of blocks read.
	 * @param unit The unit of data.
	 * @param normalize Whether or not to normalize the data.
	 * @param verbose Whether or not to output loading progress info.
	 * @return The array containing the file data.
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public double[][][] getDataFromBinary(String filePath, int fileOffset,
			int blockLength, int maxBlocks, String unit, boolean normalize, 
			boolean verbose) 
			throws IOException, InterruptedException {
		File file = new File(filePath);
		BufferedInputStream fileStream = 
			new BufferedInputStream(new FileInputStream(file));
		int dataBlocks = Math.min(
			this.getDataBlocks(file, fileOffset, blockLength),
			maxBlocks);
		double data[][][] = new double[dataBlocks][][];
		String currentProgressBar = "";
		
		fileStream.skip(fileOffset);
		
		System.out.println(START_MESSAGE + " " + file.getName() + "...");
		
		for (int i = 0; i < dataBlocks; i++) {
			double[][] block = new double[blockLength][];
			
			if (verbose) {
				if (i > 0) {
					deleteProgressBar(currentProgressBar);
				}
				
				currentProgressBar = 
					getProgressBar(PROGRESS_MESSAGE, i + 1, dataBlocks,
						unit);
				
				System.out.print(currentProgressBar);
				
				if (i == dataBlocks - 1) {
					System.out.print('\n');
				}
			}
			
			for (int j = 0; j < blockLength; j++) {
				double[] row = new double[1];
				
				row[0] = (normalize ? 
					normalize(fileStream.read(), MAX_BYTE_VALUE) : 
					fileStream.read());
				block[j] = row;
			}
			
			data[i] = block;
		}
		
		System.out.println(DONE_MESSAGE + "\n");
		
		fileStream.close();
		
		return data;
	}
	
	/**
	 * Deletes the current progress bar.
	 * @param currentProgressBar The current progress bar.
	 */
	public static void deleteProgressBar(String currentProgressBar) {
		String line = "";
		
		for (int i = 0; i < currentProgressBar.length(); i++) {
			line += '\b';
		}
		
		System.out.print(line);
	}
	
	/**
	 * Returns a string representation of a progress bar. 
	 * 
	 * @param message The message to display before the progress bar.
	 * @param current The current iteration.
	 * @param total The total amount of iterations.
	 * @return the progress bar.
	 */
	public static String getProgressBar(String message, int current, int total, 
			String unit) {
		String line = message + " [";
		String endLine = "] (" + unit + " " + current + "/" + total + ")";
		double progress = (double) current / total;
		int progressBarSize = MAX_LINE_WIDTH - line.length() - 
			endLine.length() - 1;
		int currentBarSize = (int) Math.floor(progress * progressBarSize);
		
		for (int i = 0; i < progressBarSize; i++) {
			if (i + 1 <= currentBarSize) {
				line += PROGRESS_BLOCK;
			}
			else {
				line += EMPTY_PROGRESS_BLOCK;
			}
		}
		
		line += endLine;
		
		return line;
	}
	
	/**
	 * Normalizes the given value.
	 * 
	 * @param value The value to normalize.
	 * @param maxValue The maximum value on the new scale.
	 * @return The normalized value.
	 */
	public static double normalize(int value, int maxValue) {
		return (double) value / maxValue;
	}
	
	/**
	 * Returns the amount of data blocks discovered in this file.
	 * 
	 * @param file The file to read from.
	 * @param fileOffset The byte to start reading from.
	 * @param blockLength The amount of bytes in a block of data.
	 * @return the amount of data blocks discovered in this file.
	 */
	public int getDataBlocks(File file, int fileOffset, int blockLength) {
		return (int) ((file.length() - fileOffset) / blockLength);
	}
	
	/**
	 * Returns the file path.
	 * 
	 * @return the file path.
	 */
	public String getFilePath() {
		return this.filePath;
	}
	
	/**
	 * Returns the byte to start reading this file from.
	 * 
	 * @return the byte to start reading this file from.
	 */
	public long getFileOffset() {
		return this.fileOffset;
	}
	
	/**
	 * Returns the length of a block of data.
	 * 
	 * @return The length of a block of data.
	 */
	public long getBlockLength() {
		return this.blockLength;
	}
	
	/**
	 * Returns the maximum number of blocks to read.
	 * 
	 * @return the maximum number of blocks to read.
	 */
	public long getMaxBlocks() {
		return this.maxBlocks;
	}
	
	/**
	 * Returns the converted data.
	 * 
	 * @return the converted data.
	 */
	public double[][][] getData() {
		return this.data;
	}
	
	/**
	 * Returns the unit of data.
	 * 
	 * @return the unit of data.
	 */
	public String getUnit() {
		return this.unit;
	}
}
