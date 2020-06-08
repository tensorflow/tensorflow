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

package org.tensorflow.lite.support.tensorbuffer;

import java.nio.FloatBuffer;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.SupportPreconditions;

/** Represents data buffer with float values. */
public final class TensorBufferFloat extends TensorBuffer {
  private static final DataType DATA_TYPE = DataType.FLOAT32;

  /**
   * Creates a {@link TensorBufferFloat} with specified {@code shape}.
   *
   * @throws NullPointerException if {@code shape} is null.
   * @throws IllegalArgumentException if {@code shape} has non-positive elements.
   */
  TensorBufferFloat(@NonNull int[] shape) {
    super(shape);
  }

  TensorBufferFloat() {
    super();
  }

  @Override
  public DataType getDataType() {
    return DATA_TYPE;
  }

  @Override
  @NonNull
  public float[] getFloatArray() {
    buffer.rewind();
    float[] arr = new float[flatSize];

    FloatBuffer floatBuffer = buffer.asFloatBuffer();
    floatBuffer.get(arr);
    return arr;
  }

  @Override
  public float getFloatValue(int absIndex) {
    return buffer.getFloat(absIndex << 2);
  }

  @Override
  @NonNull
  public int[] getIntArray() {
    buffer.rewind();
    int[] arr = new int[flatSize];

    for (int i = 0; i < flatSize; i++) {
      arr[i] = (int) buffer.getFloat();
    }
    return arr;
  }

  @Override
  public int getIntValue(int absIndex) {
    return (int) buffer.getFloat(absIndex << 2);
  }

  @Override
  public int getTypeSize() {
    return DATA_TYPE.byteSize();
  }

  @Override
  public void loadArray(@NonNull float[] src, @NonNull int[] shape) {
    SupportPreconditions.checkNotNull(src, "The array to be loaded cannot be null.");
    SupportPreconditions.checkArgument(
        src.length == computeFlatSize(shape),
        "The size of the array to be loaded does not match the specified shape.");
    resize(shape);
    buffer.rewind();

    FloatBuffer floatBuffer = buffer.asFloatBuffer();
    floatBuffer.put(src);
  }

  @Override
  public void loadArray(@NonNull int[] src, @NonNull int[] shape) {
    SupportPreconditions.checkNotNull(src, "The array to be loaded cannot be null.");
    SupportPreconditions.checkArgument(
        src.length == computeFlatSize(shape),
        "The size of the array to be loaded does not match the specified shape.");
    resize(shape);
    buffer.rewind();

    for (int a : src) {
      buffer.putFloat((float) a);
    }
  }
}
