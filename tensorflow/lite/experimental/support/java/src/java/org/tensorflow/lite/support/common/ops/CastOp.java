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

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.SupportPreconditions;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

/** Casts a {@link TensorBuffer} to a specified data type. */
public class CastOp implements TensorOperator {

  private final DataType destinationType;

  /**
   * Constructs a CastOp.
   *
   * <p>Note: For only converting type for a certain {@link TensorBuffer} on-the-fly rather than in
   * a processor, please directly use {@link TensorBuffer#createFrom(TensorBuffer, DataType)}.
   *
   * <p>When this Op is executed, if the original {@link TensorBuffer} is already in {@code
   * destinationType}, the original buffer will be directly returned.
   *
   * @param destinationType: The type of the casted {@link TensorBuffer}.
   * @throws IllegalArgumentException if {@code destinationType} is neither {@link DataType#UINT8}
   * nor {@link DataType#FLOAT32}.
   */
  public CastOp(DataType destinationType) {
    SupportPreconditions.checkArgument(
        destinationType == DataType.UINT8 || destinationType == DataType.FLOAT32,
        "Destination type " + destinationType + " is not supported.");
    this.destinationType = destinationType;
  }

  @Override
  public TensorBuffer apply(TensorBuffer input) {
    if (input.getDataType() == destinationType) {
      return input;
    }
    return TensorBuffer.createFrom(input, destinationType);
  }
}
