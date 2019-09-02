package io.github.jason_vega;

import java.io.File;
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
	public static final String START_MESSAGE = "Reading from file...";
	public static final String PROCESS_BLOCK_MESSAGE = "Processing block ";
	public static final String DONE_MESSAGE = "Done.\n";
	
	public static final boolean DEFAULT_VERBOSE = true;
	
	private String filePath;
	private int fileOffset;
	private int blockLength;
	private int maxBlocks;
	
	private double[][][] data;
	
	/**
	 * Loads binary data from file.
	 * 
	 * @param filePath The path to the data file.
	 * @param fileOffset The byte to start reading at.
	 * @param blockLength The length of each block of data.
	 * @param maxBlocks An upper limit on the number of blocks read.
	 * @param verbose Whether or not to output loading progress info.
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public LoadData(String filePath, int fileOffset, int blockLength, 
			int maxBlocks, boolean verbose) throws IOException, 
			InterruptedException {
		this.filePath = filePath;
		this.fileOffset = fileOffset;
		this.blockLength = blockLength;
		this.maxBlocks = maxBlocks;
		
		this.data = getDataFromBinary(filePath, fileOffset, blockLength, 
				maxBlocks, verbose);
	}
	
	/**
	 * Loads binary data from file with the default verbose setting.
	 * 
	 * @param filePath The path to the data file.
	 * @param fileOffset The byte to start reading at.
	 * @param blockLength The length of each block of data.
	 * @param maxBlocks An upper limit on the number of blocks read.
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public LoadData(String filePath, int fileOffset, int blockLength, 
			int maxBlocks) throws IOException, InterruptedException {
		this(filePath, fileOffset, blockLength, maxBlocks, DEFAULT_VERBOSE);
	}
	
	/**
	 * Reads binary data from file and saves each block of data into a 2D 
	 * array.
	 * 
	 * @param filePath The path to the data file.
	 * @param fileOffset The byte to start reading at.
	 * @param blockLength The length of each block of data.
	 * @param maxBlocks An upper limit on the number of blocks read.
	 * @param verbose Whether or not to output loading progress info.
	 * @return
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public double[][][] getDataFromBinary(String filePath, int fileOffset,
			int blockLength, int maxBlocks, boolean verbose) 
			throws IOException, InterruptedException {
		File file = new File(filePath);
		FileInputStream fileStream = new FileInputStream(file);
		int dataBlocks = Math.min(
			this.getDataBlocks(file, fileOffset, blockLength),
			maxBlocks);
		
		fileStream.skip(fileOffset);
		
		double data[][][] = new double[dataBlocks][][];
		
		System.out.println(START_MESSAGE);
		
		for (int i = 0; i < dataBlocks; i++) {
			double[][] block = new double[blockLength][];
			
			if (verbose) {
				System.out.println(PROCESS_BLOCK_MESSAGE + (i + 1) + "/" + 
					dataBlocks);
			}
			
			for (int j = 0; j < blockLength; j++) {
				double[] row = new double[1];
				
				row[0] = fileStream.read();
				block[j] = row;
			}
			
			data[i] = block;
		}
		
		System.out.println(DONE_MESSAGE);
		
		fileStream.close();
		
		return data;
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
}
