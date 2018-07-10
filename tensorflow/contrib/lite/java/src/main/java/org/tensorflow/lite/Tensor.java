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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * A typed multi-dimensional array used in Tensorflow Lite.
 *
 * <p>The native handle of a {@code Tensor} belongs to {@code NativeInterpreterWrapper}, thus not
 * needed to be closed here.
 */
final class Tensor {

  static Tensor fromHandle(long nativeHandle) {
    return new Tensor(nativeHandle);
  }

  /** Returns the {@link DataType} of elements stored in the Tensor. */
  public DataType dataType() {
    return dtype;
  }

  /** Returns the size, in bytes, of the tensor data. */
  public int numBytes() {
    return numBytes(nativeHandle);
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
    int[] inputShape = NativeInterpreterWrapper.shapeOf(input);
    if (Arrays.equals(shapeCopy, inputShape)) {
      return null;
    }
    return inputShape;
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
    DataType oType = NativeInterpreterWrapper.dataTypeOf(o);
    if (oType != dtype) {
      throw new IllegalArgumentException(
          String.format(
              "Cannot convert between a TensorFlowLite tensor with type %s and a Java "
                  + "object of type %s (which is compatible with the TensorFlowLite type %s).",
              dtype, o.getClass().getName(), oType));
    }

    int[] oShape = NativeInterpreterWrapper.shapeOf(o);
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
  private final int[] shapeCopy;

  private Tensor(long nativeHandle) {
    this.nativeHandle = nativeHandle;
    this.dtype = DataType.fromNumber(dtype(nativeHandle));
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
