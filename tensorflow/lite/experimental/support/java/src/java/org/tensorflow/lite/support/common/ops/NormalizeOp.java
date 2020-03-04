/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite.support.common.ops;

import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.SupportPreconditions;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat;

/**
 * Normalizes a {@link TensorBuffer} with given mean and stddev: output = (input - mean) / stddev.
 */
public class NormalizeOp implements TensorOperator {

  // mean.length should always be equal to stddev.length and always >= 1.
  private final float[] mean;
  private final float[] stddev;
  private final int numChannels;
  private final boolean isIdentityOp;

  /**
   * Initializes a NormalizeOp. When being called, it creates a new {@link TensorBuffer}, which
   * satisfies:
   *
   * <pre>
   *   output = (input - mean) / stddev
   * </pre>
   *
   * <p>In the following two cases, reset {@code mean} to 0 and {@code stddev} to 1 to bypass the
   * normalization. <br>
   * 1. Both {@code mean} and {code stddev} are 0. <br>
   * 2. {@code mean} is 0 and {stddev} is Infinity.
   *
   * <p>Note: If {@code mean} is set to 0 and {@code stddev} is set to 1, no computation will
   * happen, and original input will be directly returned in execution.
   *
   * <p>Note: The returned {@link TensorBuffer} is always a {@link DataType#FLOAT32} tensor at
   * present, except that the input is a {@link DataType#UINT8} tensor, {@code mean} is set to 0 and
   * {@code stddev} is set to 1.
   *
   * @param mean the mean value to be subtracted first.
   * @param stddev the standard deviation value to divide then.
   * @throws IllegalArgumentException if {@code stddev} is zero.
   */
  public NormalizeOp(float mean, float stddev) {
    // Make exceptions to the cases that
    // 1. Both mean and stddev are 0.0f. This may happen when reading the normalization parameters
    // from a tensor which does not have the values populated in the metadata. The same situation
    // may also happen to the quantization parameters.
    // 2. mean is 0.0f and stddev is Infinity. This may happen when reading the quantization
    // parameters from a tensor which does not have the values populated in the metadata, and then
    // passing the parameters into the DequantizeOp.
    // Bypass both of the two cases, by reseting stddev to 1.0f.
    if (mean == 0.0f && (stddev == 0.0f || Float.isInfinite(stddev))) {
      stddev = 1.0f;
    }

    SupportPreconditions.checkArgument(stddev != 0.0f, "Stddev cannot be zero.");
    boolean meansIsZeroAndDevsIs1 = false;
    if (mean == 0.0f && stddev == 1.0f) {
      meansIsZeroAndDevsIs1 = true;
    }

    this.isIdentityOp = meansIsZeroAndDevsIs1;
    this.mean = new float[] {mean};
    this.stddev = new float[] {stddev};
    this.numChannels = 1;
  }

  /**
   * Initializes a NormalizeOp. When being called, it creates a new {@link TensorBuffer}, which
   * satisfies:
   *
   * <pre>
   *   // Pseudo code. [...][i] means a certain element whose channel id is i.
   *   output[...][i] = (input[...][i] - mean[i]) / stddev[i]
   * </pre>
   *
   * <p>Note: If all values in {@code mean} are set to 0 and all {@code stddev} are set to 1, no
   * computation will happen, and original input will be directly returned in execution.
   *
   * <p>Note: The returned {@link TensorBuffer} is always a {@link DataType#FLOAT32} tensor at
   * present, except that the input is a {@link DataType#UINT8} tensor, all {@code mean} are set to
   * 0 and all {@code stddev} are set to 1.
   *
   * @param mean the mean values to be subtracted first for each channel.
   * @param stddev the standard deviation values to divide then for each channel.
   * @throws IllegalArgumentException if any {@code stddev} is zero, or {@code mean} has different
   *     number of elements with {@code stddev}, or any of them is empty.
   */
  public NormalizeOp(@NonNull float[] mean, @NonNull float[] stddev) {
    SupportPreconditions.checkNotNull(mean, "Mean cannot be null");
    SupportPreconditions.checkNotNull(stddev, "Stddev cannot be null");
    SupportPreconditions.checkArgument(
        mean.length == stddev.length,
        "Per channel normalization requires same number of means and stddevs");
    SupportPreconditions.checkArgument(mean.length > 0, "Means and stddevs are empty.");
    this.mean = mean.clone();
    this.stddev = stddev.clone();
    boolean allMeansAreZeroAndAllDevsAre1 = true;
    this.numChannels = mean.length;
    for (int i = 0; i < numChannels; i++) {
      SupportPreconditions.checkArgument(this.stddev[i] != 0, "Stddev cannot be zero.");
      if (this.stddev[i] != 1 || this.mean[i] != 0) {
        allMeansAreZeroAndAllDevsAre1 = false;
      }
    }
    this.isIdentityOp = allMeansAreZeroAndAllDevsAre1;
  }

  /**
   * Applies the defined normalization on given tensor and returns the result.
   *
   * <p>Note: {@code input} is possibly the same instance with the output.
   *
   * @param input input tensor. It may be the same instance with the output.
   * @return output tensor.
   */
  @Override
  @NonNull
  public TensorBuffer apply(@NonNull TensorBuffer input) {
    if (isIdentityOp) {
      return input;
    }
    int[] shape = input.getShape();
    SupportPreconditions.checkArgument(
        numChannels == 1 || (shape.length != 0 && shape[shape.length - 1] == numChannels),
        "Number of means (stddevs) is not same with number of channels (size of last axis).");
    // TODO(136750944): Eliminate the array copy here.
    float[] values = input.getFloatArray();
    int j = 0;
    for (int i = 0; i < values.length; i++) {
      values[i] = (values[i] - mean[j]) / stddev[j];
      j = (j + 1) % numChannels;
    }
    TensorBuffer output;
    if (input.isDynamic()) {
      output = TensorBufferFloat.createDynamic(DataType.FLOAT32);
    } else {
      output = TensorBufferFloat.createFixedSize(shape, DataType.FLOAT32);
    }
    output.loadArray(values, shape);
    return output;
  }
}
