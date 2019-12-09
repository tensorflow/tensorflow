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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.SupportPreconditions;

/** Represents the data buffer for either a model's input or its output. */
public abstract class TensorBuffer {
  /** Where the data is stored. */
  protected ByteBuffer buffer;

  /** Shape of the tensor stored in this buffer. */
  protected int[] shape;

  /** Number of elements in the buffer. It will be changed to a proper value in the constructor. */
  protected int flatSize = -1;

  /**
   * Indicator of whether this buffer is dynamic or fixed-size. Fixed-size buffers will have
   * pre-allocated memory and fixed size. While the size of dynamic buffers can be changed.
   */
  protected final boolean isDynamic;

  /**
   * Creates a {@link TensorBuffer} with specified {@code shape} and {@link DataType}. Here are some
   * examples:
   *
   * <pre>
   * Creating a float TensorBuffer with shape {2, 3}:
   * int[] shape = new int[] {2, 3};
   * TensorBuffer tensorBuffer = TensorBuffer.createFixedSize(shape, DataType.FLOAT32);
   * </pre>
   *
   * <pre>
   * Creating an uint8 TensorBuffer of a scalar:
   * int[] shape = new int[] {};
   * TensorBuffer tensorBuffer = TensorBuffer.createFixedSize(shape, DataType.UINT8);
   * </pre>
   *
   * <pre>
   * Creating an empty uint8 TensorBuffer:
   * int[] shape = new int[] {0};
   * TensorBuffer tensorBuffer = TensorBuffer.createFixedSize(shape, DataType.UINT8);
   * </pre>
   *
   * <p>The size of a fixed-size TensorBuffer cannot be changed once it is created.
   *
   * @param shape The shape of the {@link TensorBuffer} to be created.
   * @param dataType The dataType of the {@link TensorBuffer} to be created.
   * @throws NullPointerException if {@code shape} is null.
   * @throws IllegalArgumentException if {@code shape} has non-positive elements.
   */
  @NonNull
  public static TensorBuffer createFixedSize(@NonNull int[] shape, DataType dataType) {
    switch (dataType) {
      case FLOAT32:
        return new TensorBufferFloat(shape);
      case UINT8:
        return new TensorBufferUint8(shape);
      default:
        throw new AssertionError("TensorBuffer does not support data type: " + dataType);
    }
  }

  /**
   * Creates an empty dynamic {@link TensorBuffer} with specified {@link DataType}. The shape of the
   * created {@link TensorBuffer} is {0}.
   *
   * <p>Dynamic TensorBuffers will reallocate memory when loading arrays or data buffers of
   * different buffer sizes.
   *
   * @param dataType The dataType of the {@link TensorBuffer} to be created.
   */
  @NonNull
  public static TensorBuffer createDynamic(DataType dataType) {
    switch (dataType) {
      case FLOAT32:
        return new TensorBufferFloat();
      case UINT8:
        return new TensorBufferUint8();
      default:
        throw new AssertionError("TensorBuffer does not support data type: " + dataType);
    }
  }

  /**
   * Creates a {@link TensorBuffer} deep-copying data from another, with specified {@link DataType}.
   *
   * @param buffer the source {@link TensorBuffer} to copy from.
   * @param dataType the expected {@link DataType} of newly created {@link TensorBuffer}.
   * @throws NullPointerException if {@code buffer} is null.
   */
  @NonNull
  public static TensorBuffer createFrom(@NonNull TensorBuffer buffer, DataType dataType) {
    SupportPreconditions.checkNotNull(buffer, "Cannot create a buffer from null");
    TensorBuffer result;
    if (buffer.isDynamic()) {
      result = createDynamic(dataType);
    } else {
      result = createFixedSize(buffer.shape, dataType);
    }
    // The only scenario we need float array is FLOAT32->FLOAT32, or we can always use INT as
    // intermediate container.
    // The assumption is not true when we support other data types.
    if (buffer.getDataType() == DataType.FLOAT32 && dataType == DataType.FLOAT32) {
      float[] data = buffer.getFloatArray();
      result.loadArray(data, buffer.shape);
    } else {
      int[] data = buffer.getIntArray();
      result.loadArray(data, buffer.shape);
    }
    return result;
  }

  /** Returns the data buffer. */
  @NonNull
  public ByteBuffer getBuffer() {
    return buffer;
  }

  /** Gets the {@link TensorBuffer#flatSize} of the buffer. */
  public int getFlatSize() {
    return flatSize;
  }

  /** Gets the current shape. (returning a copy here to avoid unexpected modification.) */
  @NonNull
  public int[] getShape() {
    return Arrays.copyOf(shape, shape.length);
  }

  /** Returns the data type of this buffer. */
  public abstract DataType getDataType();

  /**
   * Returns a float array of the values stored in this buffer. If the buffer is of different types
   * than float, the values will be converted into float. For example, values in {@link
   * TensorBufferUint8} will be converted from uint8 to float.
   */
  @NonNull
  public abstract float[] getFloatArray();

  /**
   * Returns an int array of the values stored in this buffer. If the buffer is of different type
   * than int, the values will be converted into int, and loss of precision may apply. For example,
   * getting an int array from a {@link TensorBufferFloat} with values {400.32f, 23.04f}, the output
   * is {400, 23}.
   */
  @NonNull
  public abstract int[] getIntArray();

  /**
   * Returns the number of bytes of a single element in the array. For example, a float buffer will
   * return 4, and a byte buffer will return 1.
   */
  public abstract int getTypeSize();

  /** Returns if the TensorBuffer is dynamic sized (could resize arbitrarily). */
  public boolean isDynamic() {
    return isDynamic;
  }

  /**
   * Loads an int array into this buffer with specific shape. If the buffer is of different types
   * than int, the values will be converted into the buffer's type before being loaded into the
   * buffer, and loss of precision may apply. For example, loading an int array with values {400,
   * -23} into a {@link TensorBufferUint8} , the values will be clamped to [0, 255] and then be
   * casted to uint8 by {255, 0}.
   *
   * @param src The source array to be loaded.
   * @param shape Shape of the tensor that {@code src} represents.
   * @throws NullPointerException if {@code src} is null.
   * @throws NullPointerException if {@code shape} is null.
   * @throws IllegalArgumentException if the size of the array to be loaded does not match the
   *     specified shape.
   */
  public abstract void loadArray(@NonNull int[] src, @NonNull int[] shape);

  /**
   * Loads an int array into this buffer. If the buffer is of different types than int, the values
   * will be converted into the buffer's type before being loaded into the buffer, and loss of
   * precision may apply. For example, loading an int array with values {400, -23} into a {@link
   * TensorBufferUint8} , the values will be clamped to [0, 255] and then be casted to uint8 by
   * {255, 0}.
   *
   * <p>Size of {@code src} should always match the flat size of this {@link TensorBuffer}, for both
   * fixed-size and dynamic {@link TensorBuffer}.
   *
   * @param src The source array to be loaded.
   */
  public void loadArray(@NonNull int[] src) {
    loadArray(src, shape);
  }

  /**
   * Loads a float array into this buffer with specific shape. If the buffer is of different types
   * than float, the values will be converted into the buffer's type before being loaded into the
   * buffer, and loss of precision may apply. For example, loading a float array into a {@link
   * TensorBufferUint8} with values {400.32f, -23.04f}, the values will be clamped to [0, 255] and
   * then be casted to uint8 by {255, 0}.
   *
   * @param src The source array to be loaded.
   * @param shape Shape of the tensor that {@code src} represents.
   * @throws NullPointerException if {@code src} is null.
   * @throws NullPointerException if {@code shape} is null.
   * @throws IllegalArgumentException if the size of the array to be loaded does not match the
   *     specified shape.
   */
  public abstract void loadArray(@NonNull float[] src, @NonNull int[] shape);

  /**
   * Loads a float array into this buffer. If the buffer is of different types than float, the
   * values will be converted into the buffer's type before being loaded into the buffer, and loss
   * of precision may apply. For example, loading a float array into a {@link TensorBufferUint8}
   * with values {400.32f, -23.04f}, the values will be clamped to [0, 255] and then be casted to
   * uint8 by {255, 0}.
   *
   * <p>Size of {@code src} should always match the flat size of this {@link TensorBuffer}, for both
   * fixed-size and dynamic {@link TensorBuffer}.
   *
   * @param src The source array to be loaded.
   */
  public void loadArray(@NonNull float[] src) {
    loadArray(src, shape);
  }

  /**
   * Loads a byte buffer into this {@link TensorBuffer} with specific shape.
   *
   * <p>Important: The loaded buffer is a reference. DO NOT MODIFY. We don't create a copy here for
   * performance concern, but if modification is necessary, please make a copy.
   *
   * @param buffer The byte buffer to load.
   * @throws NullPointerException if {@code buffer} is null.
   * @throws IllegalArgumentException if the size of {@code buffer} and {@code typeSize} do not
   *     match or the size of {@code buffer} and {@code flatSize} do not match.
   */
  public void loadBuffer(@NonNull ByteBuffer buffer, @NonNull int[] shape) {
    SupportPreconditions.checkNotNull(buffer, "Byte buffer cannot be null.");
    int flatSize = computeFlatSize(shape);
    SupportPreconditions.checkArgument(
        (buffer.limit() == getTypeSize() * flatSize),
        "The size of byte buffer and the shape do not match.");

    if (!isDynamic) {
      SupportPreconditions.checkArgument(
          flatSize == this.flatSize,
          "The size of byte buffer and the size of the tensor buffer do not match.");
    } else {
      this.flatSize = flatSize;
    }

    this.shape = shape.clone();
    buffer.rewind();
    this.buffer = buffer;
  }

  /**
   * Loads a byte buffer into this {@link TensorBuffer}. Buffer size must match the flat size of
   * this {@link TensorBuffer}.
   *
   * <p>Important: The loaded buffer is a reference. DO NOT MODIFY. We don't create a copy here for
   * performance concern, but if modification is necessary, please make a copy.
   *
   * @param buffer The byte buffer to load.
   */
  public void loadBuffer(@NonNull ByteBuffer buffer) {
    loadBuffer(buffer, shape);
  }

  /**
   * Constructs a fixed size {@link TensorBuffer} with specified {@code shape}.
   *
   * @throws NullPointerException if {@code shape} is null.
   * @throws IllegalArgumentException if {@code shape} has non-positive elements.
   */
  protected TensorBuffer(@NonNull int[] shape) {
    isDynamic = false;
    allocateMemory(shape);
  }

  /** Constructs a dynamic {@link TensorBuffer} which can be resized. */
  protected TensorBuffer() {
    isDynamic = true;
    // Initialize the dynamic TensorBuffer with an empty ByteBuffer.
    allocateMemory(new int[] {0});
  }

  /** Calculates number of elements in the buffer. */
  protected static int computeFlatSize(@NonNull int[] shape) {
    SupportPreconditions.checkNotNull(shape, "Shape cannot be null.");
    int prod = 1;
    for (int s : shape) {
      prod = prod * s;
    }
    return prod;
  }

  /**
   * For dynamic buffer, resize the memory if needed. For fixed-size buffer, check if the {@code
   * shape} of src fits the buffer size.
   */
  protected void resize(@NonNull int[] shape) {
    if (isDynamic) {
      allocateMemory(shape);
    } else {
      // Make sure the new shape fits the buffer size when TensorBuffer has fixed size.
      SupportPreconditions.checkArgument(Arrays.equals(shape, this.shape));
      this.shape = shape.clone();
    }
  }

  /**
   * Allocates buffer with corresponding size of the {@code shape}. If shape is an empty array, this
   * TensorBuffer will be created as a scalar and its flatSize will be 1.
   *
   * @throws NullPointerException if {@code shape} is null.
   * @throws IllegalArgumentException if {@code shape} has negative elements.
   */
  private void allocateMemory(@NonNull int[] shape) {
    SupportPreconditions.checkNotNull(shape, "TensorBuffer shape cannot be null.");
    SupportPreconditions.checkArgument(
        isShapeValid(shape), "Values in TensorBuffer shape should be non-negative.");

    // Check if the new shape is the same as current shape.
    int newFlatSize = computeFlatSize(shape);
    if (flatSize == newFlatSize) {
      return;
    }

    // Update to the new shape.
    flatSize = newFlatSize;
    this.shape = shape.clone();
    buffer = ByteBuffer.allocateDirect(flatSize * getTypeSize());
    buffer.order(ByteOrder.nativeOrder());
  }

  /**
   * Checks if {@code shape} meets one of following two requirements: 1. Elements in {@code shape}
   * are all non-negative numbers. 2. {@code shape} is an empty array, which corresponds to scalar.
   */
  private static boolean isShapeValid(@NonNull int[] shape) {
    if (shape.length == 0) {
      // This shape refers to a scalar.
      return true;
    }

    // This shape refers to a multidimentional array.
    for (int s : shape) {
      // All elements in shape should be non-negative.
      if (s < 0) {
        return false;
      }
    }
    return true;
  }
}
