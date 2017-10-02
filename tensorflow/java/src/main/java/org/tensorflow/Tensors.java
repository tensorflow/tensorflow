package org.tensorflow;

import static java.nio.charset.StandardCharsets.UTF_8;

/** Utility class offering convenience methods for creating Tensors in a type-safe way. */
public class Tensors {
  private Tensors() {}

  /** Creates a scalar String tensor using the default, UTF-8 encoding. */
  public static Tensor<String> create(String data) {
    return Tensor.create(data.getBytes(UTF_8), String.class);
  }

  /**
   * Creates a scalar String tensor using a specified encoding.
   *
   * @param charset The encoding from String to bytes.
   */
  public static Tensor<String> create(String data, java.nio.charset.Charset charset) {
    return Tensor.create(data.getBytes(charset), String.class);
  }

  /**
   * Creates a scalar tensor containing a single Float element.
   *
   * @param data The value to put into the new scalar tensor.
   */
  public static Tensor<Float> create(float data) {
    return Tensor.create(data, Float.class);
  }

  /**
   * Creates a rank-1 tensor of Float elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Float> create(float[] data) {
    return Tensor.create(data, Float.class);
  }

  /**
   * Creates a rank-2 tensor of Float elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Float> create(float[][] data) {
    return Tensor.create(data, Float.class);
  }

  /**
   * Creates a rank-3 tensor of Float elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Float> create(float[][][] data) {
    return Tensor.create(data, Float.class);
  }

  /**
   * Creates a rank-4 tensor of Float elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Float> create(float[][][][] data) {
    return Tensor.create(data, Float.class);
  }

  /**
   * Creates a rank-5 tensor of Float elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Float> create(float[][][][][] data) {
    return Tensor.create(data, Float.class);
  }

  /**
   * Creates a rank-6 tensor of Float elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Float> create(float[][][][][][] data) {
    return Tensor.create(data, Float.class);
  }

  /**
   * Creates a scalar tensor containing a single Double element.
   *
   * @param data The value to put into the new scalar tensor.
   */
  public static Tensor<Double> create(double data) {
    return Tensor.create(data, Double.class);
  }

  /**
   * Creates a rank-1 tensor of Double elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Double> create(double[] data) {
    return Tensor.create(data, Double.class);
  }

  /**
   * Creates a rank-2 tensor of Double elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Double> create(double[][] data) {
    return Tensor.create(data, Double.class);
  }

  /**
   * Creates a rank-3 tensor of Double elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Double> create(double[][][] data) {
    return Tensor.create(data, Double.class);
  }

  /**
   * Creates a rank-4 tensor of Double elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Double> create(double[][][][] data) {
    return Tensor.create(data, Double.class);
  }

  /**
   * Creates a rank-5 tensor of Double elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Double> create(double[][][][][] data) {
    return Tensor.create(data, Double.class);
  }

  /**
   * Creates a rank-6 tensor of Double elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Double> create(double[][][][][][] data) {
    return Tensor.create(data, Double.class);
  }

  /**
   * Creates a scalar tensor containing a single Integer element.
   *
   * @param data The value to put into the new scalar tensor.
   */
  public static Tensor<Integer> create(int data) {
    return Tensor.create(data, Integer.class);
  }

  /**
   * Creates a rank-1 tensor of Integer elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Integer> create(int[] data) {
    return Tensor.create(data, Integer.class);
  }

  /**
   * Creates a rank-2 tensor of Integer elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Integer> create(int[][] data) {
    return Tensor.create(data, Integer.class);
  }

  /**
   * Creates a rank-3 tensor of Integer elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Integer> create(int[][][] data) {
    return Tensor.create(data, Integer.class);
  }

  /**
   * Creates a rank-4 tensor of Integer elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Integer> create(int[][][][] data) {
    return Tensor.create(data, Integer.class);
  }

  /**
   * Creates a rank-5 tensor of Integer elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Integer> create(int[][][][][] data) {
    return Tensor.create(data, Integer.class);
  }

  /**
   * Creates a rank-6 tensor of Integer elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Integer> create(int[][][][][][] data) {
    return Tensor.create(data, Integer.class);
  }

  /**
   * Creates a scalar tensor containing a single String element.
   *
   * @param data An array containing the data to put into the new tensor. String elements are
   *     sequences of bytes from the last array dimension.
   */
  public static Tensor<String> create(byte[] data) {
    return Tensor.create(data, String.class);
  }

  /**
   * Creates a rank-1 tensor of String elements.
   *
   * @param data An array containing the data to put into the new tensor. String elements are
   *     sequences of bytes from the last array dimension.
   */
  public static Tensor<String> create(byte[][] data) {
    return Tensor.create(data, String.class);
  }

  /**
   * Creates a rank-2 tensor of String elements.
   *
   * @param data An array containing the data to put into the new tensor. String elements are
   *     sequences of bytes from the last array dimension.
   */
  public static Tensor<String> create(byte[][][] data) {
    return Tensor.create(data, String.class);
  }

  /**
   * Creates a rank-3 tensor of String elements.
   *
   * @param data An array containing the data to put into the new tensor. String elements are
   *     sequences of bytes from the last array dimension.
   */
  public static Tensor<String> create(byte[][][][] data) {
    return Tensor.create(data, String.class);
  }

  /**
   * Creates a rank-4 tensor of String elements.
   *
   * @param data An array containing the data to put into the new tensor. String elements are
   *     sequences of bytes from the last array dimension.
   */
  public static Tensor<String> create(byte[][][][][] data) {
    return Tensor.create(data, String.class);
  }

  /**
   * Creates a rank-5 tensor of String elements.
   *
   * @param data An array containing the data to put into the new tensor. String elements are
   *     sequences of bytes from the last array dimension.
   */
  public static Tensor<String> create(byte[][][][][][] data) {
    return Tensor.create(data, String.class);
  }

  /**
   * Creates a scalar tensor containing a single Long element.
   *
   * @param data The value to put into the new scalar tensor.
   */
  public static Tensor<Long> create(long data) {
    return Tensor.create(data, Long.class);
  }

  /**
   * Creates a rank-1 tensor of Long elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Long> create(long[] data) {
    return Tensor.create(data, Long.class);
  }

  /**
   * Creates a rank-2 tensor of Long elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Long> create(long[][] data) {
    return Tensor.create(data, Long.class);
  }

  /**
   * Creates a rank-3 tensor of Long elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Long> create(long[][][] data) {
    return Tensor.create(data, Long.class);
  }

  /**
   * Creates a rank-4 tensor of Long elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Long> create(long[][][][] data) {
    return Tensor.create(data, Long.class);
  }

  /**
   * Creates a rank-5 tensor of Long elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Long> create(long[][][][][] data) {
    return Tensor.create(data, Long.class);
  }

  /**
   * Creates a rank-6 tensor of Long elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Long> create(long[][][][][][] data) {
    return Tensor.create(data, Long.class);
  }

  /**
   * Creates a scalar tensor containing a single Boolean element.
   *
   * @param data The value to put into the new scalar tensor.
   */
  public static Tensor<Boolean> create(boolean data) {
    return Tensor.create(data, Boolean.class);
  }

  /**
   * Creates a rank-1 tensor of Boolean elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Boolean> create(boolean[] data) {
    return Tensor.create(data, Boolean.class);
  }

  /**
   * Creates a rank-2 tensor of Boolean elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Boolean> create(boolean[][] data) {
    return Tensor.create(data, Boolean.class);
  }

  /**
   * Creates a rank-3 tensor of Boolean elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Boolean> create(boolean[][][] data) {
    return Tensor.create(data, Boolean.class);
  }

  /**
   * Creates a rank-4 tensor of Boolean elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Boolean> create(boolean[][][][] data) {
    return Tensor.create(data, Boolean.class);
  }

  /**
   * Creates a rank-5 tensor of Boolean elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Boolean> create(boolean[][][][][] data) {
    return Tensor.create(data, Boolean.class);
  }

  /**
   * Creates a rank-6 tensor of Boolean elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Boolean> create(boolean[][][][][][] data) {
    return Tensor.create(data, Boolean.class);
  }
}
