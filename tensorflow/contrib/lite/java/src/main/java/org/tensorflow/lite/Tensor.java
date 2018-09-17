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

package org.tensorflow.lite;

import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * A typed multi-dimensional array used in Tensorflow Lite.
 *
 * <p>The native handle of a {@code Tensor} belongs to {@code NativeInterpreterWrapper}, thus not
 * needed to be closed here.
 */
public final class Tensor {

  static Tensor fromHandle(long nativeHandle) {
    return new Tensor(nativeHandle);
  }

  /** Returns the {@link DataType} of elements stored in the Tensor. */
  public DataType dataType() {
    return dtype;
  }

  /**
   * Returns the number of dimensions (sometimes referred to as <a
   * href="https://www.tensorflow.org/resources/dims_types.html#rank">rank</a>) of the Tensor.
   *
   * <p>Will be 0 for a scalar, 1 for a vector, 2 for a matrix, 3 for a 3-dimensional tensor etc.
   */
  public int numDimensions() {
    return shapeCopy.length;
  }

  /** Returns the size, in bytes, of the tensor data. */
  public int numBytes() {
    return numBytes(nativeHandle);
  }

  /** Returns the number of elements in a flattened (1-D) view of the tensor. */
  public int numElements() {
    return computeNumElements(shapeCopy);
  }

  /**
   * Returns the <a href="https://www.tensorflow.org/resources/dims_types.html#shape">shape</a> of
   * the Tensor, i.e., the sizes of each dimension.
   *
   * @return an array where the i-th element is the size of the i-th dimension of the tensor.
   */
  public int[] shape() {
    return shapeCopy;
  }

  /**
   * Copies the contents of the provided {@code src} object to the Tensor.
   *
   * <p>The {@code src} should either be a (multi-dimensional) array with a shape matching that of
   * this tensor, or a {@link ByteByffer} of compatible primitive type with a matching flat size.
   *
   * @throws IllegalArgumentException if the tensor is a scalar or if {@code src} is not compatible
   *     with the tensor (for example, mismatched data types or shapes).
   */
  void setTo(Object src) {
    throwExceptionIfTypeIsIncompatible(src);
    if (isByteBuffer(src)) {
      ByteBuffer srcBuffer = (ByteBuffer) src;
      // For direct ByteBuffer instances we support zero-copy. Note that this assumes the caller
      // retains ownership of the source buffer until inference has completed.
      if (srcBuffer.isDirect() && srcBuffer.order() == ByteOrder.nativeOrder()) {
        writeDirectBuffer(nativeHandle, srcBuffer);
      } else {
        buffer().put(srcBuffer);
      }
      return;
    }
    writeMultiDimensionalArray(nativeHandle, src);
  }

  /**
   * Copies the contents of the tensor to {@code dst} and returns {@code dst}.
   *
   * @param dst the destination buffer, either an explicitly-typed array or a {@link ByteBuffer}.
   * @throws IllegalArgumentException if {@code dst} is not compatible with the tensor (for example,
   *     mismatched data types or shapes).
   */
  Object copyTo(Object dst) {
    throwExceptionIfTypeIsIncompatible(dst);
    if (dst instanceof ByteBuffer) {
      ByteBuffer dstByteBuffer = (ByteBuffer) dst;
      dstByteBuffer.put(buffer());
      return dst;
    }
    readMultiDimensionalArray(nativeHandle, dst);
    return dst;
  }

  /** Returns the provided buffer's shape if specified and different from this Tensor's shape. */
  // TODO(b/80431971): Remove this method after deprecating multi-dimensional array inputs.
  int[] getInputShapeIfDifferent(Object input) {
    // Implicit resizes based on ByteBuffer capacity isn't supported, so short-circuit that path.
    // The ByteBuffer's size will be validated against this Tensor's size in {@link #setTo(Object)}.
    if (isByteBuffer(input)) {
      return null;
    }
    int[] inputShape = computeShapeOf(input);
    if (Arrays.equals(shapeCopy, inputShape)) {
      return null;
    }
    return inputShape;
  }

  /**
   * Forces a refresh of the tensor's cached shape.
   *
   * <p>This is useful if the tensor is resized or has a dynamic shape.
   */
  void refreshShape() {
    this.shapeCopy = shape(nativeHandle);
  }

  /** Returns the type of the data. */
  static DataType dataTypeOf(Object o) {
    if (o != null) {
      Class<?> c = o.getClass();
      while (c.isArray()) {
        c = c.getComponentType();
      }
      if (float.class.equals(c)) {
        return DataType.FLOAT32;
      } else if (int.class.equals(c)) {
        return DataType.INT32;
      } else if (byte.class.equals(c)) {
        return DataType.UINT8;
      } else if (long.class.equals(c)) {
        return DataType.INT64;
      }
    }
    throw new IllegalArgumentException(
        "DataType error: cannot resolve DataType of " + o.getClass().getName());
  }

  /** Returns the shape of an object as an int array. */
  static int[] computeShapeOf(Object o) {
    int size = computeNumDimensions(o);
    int[] dimensions = new int[size];
    fillShape(o, 0, dimensions);
    return dimensions;
  }

  /** Returns the number of elements in a flattened (1-D) view of the tensor's shape. */
  static int computeNumElements(int[] shape) {
    int n = 1;
    for (int i = 0; i < shape.length; ++i) {
      n *= shape[i];
    }
    return n;
  }

  /** Returns the number of dimensions of a multi-dimensional array, otherwise 0. */
  static int computeNumDimensions(Object o) {
    if (o == null || !o.getClass().isArray()) {
      return 0;
    }
    if (Array.getLength(o) == 0) {
      throw new IllegalArgumentException("Array lengths cannot be 0.");
    }
    return 1 + computeNumDimensions(Array.get(o, 0));
  }

  /** Recursively populates the shape dimensions for a given (multi-dimensional) array. */
  static void fillShape(Object o, int dim, int[] shape) {
    if (shape == null || dim == shape.length) {
      return;
    }
    final int len = Array.getLength(o);
    if (shape[dim] == 0) {
      shape[dim] = len;
    } else if (shape[dim] != len) {
      throw new IllegalArgumentException(
          String.format("Mismatched lengths (%d and %d) in dimension %d", shape[dim], len, dim));
    }
    for (int i = 0; i < len; ++i) {
      fillShape(Array.get(o, i), dim + 1, shape);
    }
  }

  private void throwExceptionIfTypeIsIncompatible(Object o) {
    if (isByteBuffer(o)) {
      ByteBuffer oBuffer = (ByteBuffer) o;
      if (oBuffer.capacity() != numBytes()) {
        throw new IllegalArgumentException(
            String.format(
                "Cannot convert between a TensorFlowLite buffer with %d bytes and a "
                    + "ByteBuffer with %d bytes.",
                numBytes(), oBuffer.capacity()));
      }
      return;
    }
    DataType oType = dataTypeOf(o);
    if (oType != dtype) {
      throw new IllegalArgumentException(
          String.format(
              "Cannot convert between a TensorFlowLite tensor with type %s and a Java "
                  + "object of type %s (which is compatible with the TensorFlowLite type %s).",
              dtype, o.getClass().getName(), oType));
    }

    int[] oShape = computeShapeOf(o);
    if (!Arrays.equals(oShape, shapeCopy)) {
      throw new IllegalArgumentException(
          String.format(
              "Cannot copy between a TensorFlowLite tensor with shape %s and a Java object "
                  + "with shape %s.",
              Arrays.toString(shapeCopy), Arrays.toString(oShape)));
    }
  }

  private static boolean isByteBuffer(Object o) {
    return o instanceof ByteBuffer;
  }

  private final long nativeHandle;
  private final DataType dtype;
  private int[] shapeCopy;

  private Tensor(long nativeHandle) {
    this.nativeHandle = nativeHandle;
    this.dtype = DataType.fromC(dtype(nativeHandle));
    this.shapeCopy = shape(nativeHandle);
  }

  private ByteBuffer buffer() {
    return buffer(nativeHandle).order(ByteOrder.nativeOrder());
  }

  private static native ByteBuffer buffer(long handle);

  private static native void writeDirectBuffer(long handle, ByteBuffer src);

  private static native int dtype(long handle);

  private static native int[] shape(long handle);

  private static native int numBytes(long handle);

  private static native void readMultiDimensionalArray(long handle, Object dst);

  private static native void writeMultiDimensionalArray(long handle, Object src);

  static {
    TensorFlowLite.init();
  }
}
