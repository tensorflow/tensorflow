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
import org.tensorflow.lite.support.common.SupportPrecondtions;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat;

/**
 * Normalize a TensorBuffer with given mean and stddev: output = (input - mean) / stddev.
 */
public class NormalizeOp implements TensorOperator {

  // TODO(136750944): Support normalization on multiple channels differently.
  private final float mean;
  private final float stddev;

  /**
   * Initializes a NormalizeOp. When called, it creates a new {@link TensorBuffer}, which satisfies:
   *
   * <pre>
   *   output = (input - mean) / stddev
   * </pre>
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
    SupportPrecondtions.checkArgument(stddev != 0, "Stddev cannot be zero.");
    this.mean = mean;
    this.stddev = stddev;
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
    if (mean == 0 && stddev == 1) {
      return input;
    }
    // TODO(136750944): Eliminate the array copy here.
    int[] shape = input.getShape();
    float[] values = input.getFloatArray();
    for (int i = 0; i < values.length; i++) {
      values[i] = (values[i] - mean) / stddev;
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
