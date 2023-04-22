/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow;

import static java.nio.charset.StandardCharsets.UTF_8;

/** Type-safe factory methods for creating {@link org.tensorflow.Tensor} objects. */
public final class Tensors {
  private Tensors() {}

  /**
   * Creates a scalar String tensor using the default, UTF-8 encoding.
   *
   * @param data The string to put into the new scalar tensor.
   */
  public static Tensor<String> create(String data) {
    return Tensor.create(data.getBytes(UTF_8), String.class);
  }

  /**
   * Creates a scalar String tensor using a specified encoding.
   *
   * @param charset The encoding from String to bytes.
   * @param data The string to put into the new scalar tensor.
   */
  public static Tensor<String> create(String data, java.nio.charset.Charset charset) {
    return Tensor.create(data.getBytes(charset), String.class);
  }

  /**
   * Creates a scalar tensor containing a single {@code float} element.
   *
   * @param data The value to put into the new scalar tensor.
   */
  public static Tensor<Float> create(float data) {
    return Tensor.create(data, Float.class);
  }

  /**
   * Creates a rank-1 tensor of {@code float} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Float> create(float[] data) {
    return Tensor.create(data, Float.class);
  }

  /**
   * Creates a rank-2 tensor of {@code float} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Float> create(float[][] data) {
    return Tensor.create(data, Float.class);
  }

  /**
   * Creates a rank-3 tensor of {@code float} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Float> create(float[][][] data) {
    return Tensor.create(data, Float.class);
  }

  /**
   * Creates a rank-4 tensor of {@code float} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Float> create(float[][][][] data) {
    return Tensor.create(data, Float.class);
  }

  /**
   * Creates a rank-5 tensor of {@code float} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Float> create(float[][][][][] data) {
    return Tensor.create(data, Float.class);
  }

  /**
   * Creates a rank-6 tensor of {@code float} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Float> create(float[][][][][][] data) {
    return Tensor.create(data, Float.class);
  }

  /**
   * Creates a scalar tensor containing a single {@code double} element.
   *
   * @param data The value to put into the new scalar tensor.
   */
  public static Tensor<Double> create(double data) {
    return Tensor.create(data, Double.class);
  }

  /**
   * Creates a rank-1 tensor of {@code double} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Double> create(double[] data) {
    return Tensor.create(data, Double.class);
  }

  /**
   * Creates a rank-2 tensor of {@code double} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Double> create(double[][] data) {
    return Tensor.create(data, Double.class);
  }

  /**
   * Creates a rank-3 tensor of {@code double} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Double> create(double[][][] data) {
    return Tensor.create(data, Double.class);
  }

  /**
   * Creates a rank-4 tensor of {@code double} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Double> create(double[][][][] data) {
    return Tensor.create(data, Double.class);
  }

  /**
   * Creates a rank-5 tensor of {@code double} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Double> create(double[][][][][] data) {
    return Tensor.create(data, Double.class);
  }

  /**
   * Creates a rank-6 tensor of {@code double} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Double> create(double[][][][][][] data) {
    return Tensor.create(data, Double.class);
  }

  /**
   * Creates a scalar tensor containing a single {@code int} element.
   *
   * @param data The value to put into the new scalar tensor.
   */
  public static Tensor<Integer> create(int data) {
    return Tensor.create(data, Integer.class);
  }

  /**
   * Creates a rank-1 tensor of {@code int} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Integer> create(int[] data) {
    return Tensor.create(data, Integer.class);
  }

  /**
   * Creates a rank-2 tensor of {@code int} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Integer> create(int[][] data) {
    return Tensor.create(data, Integer.class);
  }

  /**
   * Creates a rank-3 tensor of {@code int} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Integer> create(int[][][] data) {
    return Tensor.create(data, Integer.class);
  }

  /**
   * Creates a rank-4 tensor of {@code int} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Integer> create(int[][][][] data) {
    return Tensor.create(data, Integer.class);
  }

  /**
   * Creates a rank-5 tensor of {@code int} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Integer> create(int[][][][][] data) {
    return Tensor.create(data, Integer.class);
  }

  /**
   * Creates a rank-6 tensor of {@code int} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Integer> create(int[][][][][][] data) {
    return Tensor.create(data, Integer.class);
  }

  /**
   * Creates a scalar tensor containing a single {@code byte} element.
   *
   * @param data An array containing the data to put into the new tensor. String elements are
   *     sequences of bytes from the last array dimension.
   */
  public static Tensor<String> create(byte[] data) {
    return Tensor.create(data, String.class);
  }

  /**
   * Creates a rank-1 tensor of {@code byte} elements.
   *
   * @param data An array containing the data to put into the new tensor. String elements are
   *     sequences of bytes from the last array dimension.
   */
  public static Tensor<String> create(byte[][] data) {
    return Tensor.create(data, String.class);
  }

  /**
   * Creates a rank-2 tensor of {@code byte} elements.
   *
   * @param data An array containing the data to put into the new tensor. String elements are
   *     sequences of bytes from the last array dimension.
   */
  public static Tensor<String> create(byte[][][] data) {
    return Tensor.create(data, String.class);
  }

  /**
   * Creates a rank-3 tensor of {@code byte} elements.
   *
   * @param data An array containing the data to put into the new tensor. String elements are
   *     sequences of bytes from the last array dimension.
   */
  public static Tensor<String> create(byte[][][][] data) {
    return Tensor.create(data, String.class);
  }

  /**
   * Creates a rank-4 tensor of {@code byte} elements.
   *
   * @param data An array containing the data to put into the new tensor. String elements are
   *     sequences of bytes from the last array dimension.
   */
  public static Tensor<String> create(byte[][][][][] data) {
    return Tensor.create(data, String.class);
  }

  /**
   * Creates a rank-5 tensor of {@code byte} elements.
   *
   * @param data An array containing the data to put into the new tensor. String elements are
   *     sequences of bytes from the last array dimension.
   */
  public static Tensor<String> create(byte[][][][][][] data) {
    return Tensor.create(data, String.class);
  }

  /**
   * Creates a scalar tensor containing a single {@code long} element.
   *
   * @param data The value to put into the new scalar tensor.
   */
  public static Tensor<Long> create(long data) {
    return Tensor.create(data, Long.class);
  }

  /**
   * Creates a rank-1 tensor of {@code long} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Long> create(long[] data) {
    return Tensor.create(data, Long.class);
  }

  /**
   * Creates a rank-2 tensor of {@code long} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Long> create(long[][] data) {
    return Tensor.create(data, Long.class);
  }

  /**
   * Creates a rank-3 tensor of {@code long} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Long> create(long[][][] data) {
    return Tensor.create(data, Long.class);
  }

  /**
   * Creates a rank-4 tensor of {@code long} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Long> create(long[][][][] data) {
    return Tensor.create(data, Long.class);
  }

  /**
   * Creates a rank-5 tensor of {@code long} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Long> create(long[][][][][] data) {
    return Tensor.create(data, Long.class);
  }

  /**
   * Creates a rank-6 tensor of {@code long} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Long> create(long[][][][][][] data) {
    return Tensor.create(data, Long.class);
  }

  /**
   * Creates a scalar tensor containing a single {@code boolean} element.
   *
   * @param data The value to put into the new scalar tensor.
   */
  public static Tensor<Boolean> create(boolean data) {
    return Tensor.create(data, Boolean.class);
  }

  /**
   * Creates a rank-1 tensor of {@code boolean} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Boolean> create(boolean[] data) {
    return Tensor.create(data, Boolean.class);
  }

  /**
   * Creates a rank-2 tensor of {@code boolean} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Boolean> create(boolean[][] data) {
    return Tensor.create(data, Boolean.class);
  }

  /**
   * Creates a rank-3 tensor of {@code boolean} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Boolean> create(boolean[][][] data) {
    return Tensor.create(data, Boolean.class);
  }

  /**
   * Creates a rank-4 tensor of {@code boolean} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Boolean> create(boolean[][][][] data) {
    return Tensor.create(data, Boolean.class);
  }

  /**
   * Creates a rank-5 tensor of {@code boolean} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Boolean> create(boolean[][][][][] data) {
    return Tensor.create(data, Boolean.class);
  }

  /**
   * Creates a rank-6 tensor of {@code boolean} elements.
   *
   * @param data An array containing the values to put into the new tensor. The dimensions of the
   *     new tensor will match those of the array.
   */
  public static Tensor<Boolean> create(boolean[][][][][][] data) {
    return Tensor.create(data, Boolean.class);
  }
}
