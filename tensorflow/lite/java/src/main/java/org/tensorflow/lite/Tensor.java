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
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;

/**
 * A typed multi-dimensional array used in Tensorflow Lite.
 *
 * <p>The native handle of a {@code Tensor} is managed by {@code NativeInterpreterWrapper}, and does
 * not needed to be closed by the client. However, once the {@code NativeInterpreterWrapper} has
 * been closed, the tensor handle will be invalidated.
 */
// TODO(b/153882978): Add scalar getters similar to TF's Java API.
public final class Tensor {

  /**
   * Creates a Tensor wrapper from the provided interpreter instance and tensor index.
   *
   * <p>The caller is responsible for closing the created wrapper, and ensuring the provided native
   * interpreter is valid until the tensor is closed.
   */
  static Tensor fromIndex(long nativeInterpreterHandle, int tensorIndex) {
    return new Tensor(create(nativeInterpreterHandle, tensorIndex));
  }

  /**
   * Quantization parameters that corresponds to the table, {@code QuantizationParameters}, in the
   * <a
   * href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs">TFLite
   * Model schema file.</a>
   *
   * <p>Since per-channel quantization does not apply to input and output tensors, {@code scale} and
   * {@code zero_point} are both single values instead of arrays.
   *
   * <p>For tensor that are not quantized, the values of scale and zero_point are both 0.
   *
   * <p>Given a quantized value q, the corresponding float value f should be: <br>
   * f = scale * (q - zero_point) <br>
   */
  public static class QuantizationParams {
    /** The scale value used in quantization. */
    private final float scale;
    /** The zero point value used in quantization. */
    private final int zeroPoint;

    /**
     * Creates a {@link QuantizationParams} with {@code scale} and {@code zero_point}.
     *
     * @param scale The scale value used in quantization.
     * @param zeroPoint The zero point value used in quantization.
     */
    public QuantizationParams(final float scale, final int zeroPoint) {
      this.scale = scale;
      this.zeroPoint = zeroPoint;
    }

    /** Returns the scale value. */
    public float getScale() {
      return scale;
    }

    /** Returns the zero point value. */
    public int getZeroPoint() {
      return zeroPoint;
    }
  }

  /** Disposes of any resources used by the Tensor wrapper. */
  void close() {
    delete(nativeHandle);
    nativeHandle = 0;
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
   * Returns the original <a
   * href="https://www.tensorflow.org/resources/dims_types.html#shape">shape</a> of the Tensor,
   * i.e., the sizes of each dimension - before any resizing was performed. Unknown dimensions are
   * designated with a value of -1.
   *
   * @return an array where the i-th element is the size of the i-th dimension of the tensor.
   */
  public int[] shapeSignature() {
    return shapeSignatureCopy;
  }

  /**
   * Returns the (global) index of the tensor within the owning {@link Interpreter}.
   *
   * @hide
   */
  public int index() {
    return index(nativeHandle);
  }

  /**
   * Returns the name of the tensor within the owning {@link Interpreter}.
   *
   * @hide
   */
  public String name() {
    return name(nativeHandle);
  }

  /**
   * Returns the quantization parameters of the tensor within the owning {@link Interpreter}.
   *
   * <p>Only quantized tensors have valid {@code QuantizationParameters}. For tensor that are not
   * quantized, the values of scale and zero_point are both 0.
   */
  public QuantizationParams quantizationParams() {
    return quantizationParamsCopy;
  }

  /**
   * Copies the contents of the provided {@code src} object to the Tensor.
   *
   * <p>The {@code src} should either be a (multi-dimensional) array with a shape matching that of
   * this tensor, a {@link ByteByffer} of compatible primitive type with a matching flat size, or
   * {@code null} iff the tensor has an underlying delegate buffer handle.
   *
   * @throws IllegalArgumentException if the tensor is a scalar or if {@code src} is not compatible
   *     with the tensor (for example, mismatched data types or shapes).
   */
  void setTo(Object src) {
    if (src == null) {
      if (hasDelegateBufferHandle(nativeHandle)) {
        return;
      }
      throw new IllegalArgumentException(
          "Null inputs are allowed only if the Tensor is bound to a buffer handle.");
    }
    throwIfTypeIsIncompatible(src);
    throwIfSrcShapeIsIncompatible(src);
    if (isBuffer(src)) {
      setTo((Buffer) src);
    } else if (src.getClass().isArray()) {
      writeMultiDimensionalArray(nativeHandle, src);
    } else {
      writeScalar(nativeHandle, src);
    }
  }

  private void setTo(Buffer src) {
    // Note that we attempt to use a direct memcpy optimization for direct, native-ordered buffers.
    // There are no base Buffer#order() or Buffer#put() methods, so again we have to ugly cast.
    if (src instanceof ByteBuffer) {
      ByteBuffer srcBuffer = (ByteBuffer) src;
      if (srcBuffer.isDirect() && srcBuffer.order() == ByteOrder.nativeOrder()) {
        writeDirectBuffer(nativeHandle, src);
      } else {
        buffer().put(srcBuffer);
      }
    } else if (src instanceof LongBuffer) {
      LongBuffer srcBuffer = (LongBuffer) src;
      if (srcBuffer.isDirect() && srcBuffer.order() == ByteOrder.nativeOrder()) {
        writeDirectBuffer(nativeHandle, src);
      } else {
        buffer().asLongBuffer().put(srcBuffer);
      }
    } else if (src instanceof FloatBuffer) {
      FloatBuffer srcBuffer = (FloatBuffer) src;
      if (srcBuffer.isDirect() && srcBuffer.order() == ByteOrder.nativeOrder()) {
        writeDirectBuffer(nativeHandle, src);
      } else {
        buffer().asFloatBuffer().put(srcBuffer);
      }
    } else if (src instanceof IntBuffer) {
      IntBuffer srcBuffer = (IntBuffer) src;
      if (srcBuffer.isDirect() && srcBuffer.order() == ByteOrder.nativeOrder()) {
        writeDirectBuffer(nativeHandle, src);
      } else {
        buffer().asIntBuffer().put(srcBuffer);
      }
    } else {
      throw new IllegalArgumentException("Unexpected input buffer type: " + src);
    }
  }

  /**
   * Copies the contents of the tensor to {@code dst} and returns {@code dst}.
   *
   * @param dst the destination buffer, either an explicitly-typed array, a {@link ByteBuffer} or
   *     {@code null} iff the tensor has an underlying delegate buffer handle.
   * @throws IllegalArgumentException if {@code dst} is not compatible with the tensor (for example,
   *     mismatched data types or shapes).
   */
  Object copyTo(Object dst) {
    if (dst == null) {
      if (hasDelegateBufferHandle(nativeHandle)) {
        return dst;
      }
      throw new IllegalArgumentException(
          "Null outputs are allowed only if the Tensor is bound to a buffer handle.");
    }
    throwIfTypeIsIncompatible(dst);
    throwIfDstShapeIsIncompatible(dst);
    if (isBuffer(dst)) {
      copyTo((Buffer) dst);
    } else {
      readMultiDimensionalArray(nativeHandle, dst);
    }
    return dst;
  }

  private void copyTo(Buffer dst) {
    // There is no base Buffer#put() method, so we have to ugly cast.
    if (dst instanceof ByteBuffer) {
      ((ByteBuffer) dst).put(buffer());
    } else if (dst instanceof FloatBuffer) {
      ((FloatBuffer) dst).put(buffer().asFloatBuffer());
    } else if (dst instanceof LongBuffer) {
      ((LongBuffer) dst).put(buffer().asLongBuffer());
    } else if (dst instanceof IntBuffer) {
      ((IntBuffer) dst).put(buffer().asIntBuffer());
    } else {
      throw new IllegalArgumentException("Unexpected output buffer type: " + dst);
    }
  }

  /** Returns the provided buffer's shape if specified and different from this Tensor's shape. */
  // TODO(b/80431971): Remove this method after deprecating multi-dimensional array inputs.
  int[] getInputShapeIfDifferent(Object input) {
    if (input == null) {
      return null;
    }
    // Implicit resizes based on ByteBuffer capacity isn't supported, so short-circuit that path.
    // The Buffer's size will be validated against this Tensor's size in {@link #setTo(Object)}.
    if (isBuffer(input)) {
      return null;
    }
    throwIfTypeIsIncompatible(input);
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
      // For arrays, the data elements must be a *primitive* type, e.g., an
      // array of floats is fine, but not an array of Floats.
      if (c.isArray()) {
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
        } else if (String.class.equals(c)) {
          return DataType.STRING;
        }
      } else {
        // For scalars, the type will be boxed.
        if (Float.class.equals(c) || o instanceof FloatBuffer) {
          return DataType.FLOAT32;
        } else if (Integer.class.equals(c) || o instanceof IntBuffer) {
          return DataType.INT32;
        } else if (Byte.class.equals(c)) {
          // Note that we don't check for ByteBuffer here; ByteBuffer payloads
          // are allowed to map to any type, and should be handled earlier
          // in the input/output processing pipeline.
          return DataType.UINT8;
        } else if (Long.class.equals(c) || o instanceof LongBuffer) {
          return DataType.INT64;
        } else if (String.class.equals(c)) {
          return DataType.STRING;
        }
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

  private void throwIfTypeIsIncompatible(Object o) {
    // ByteBuffer payloads can map to any type, so exempt it from the check.
    if (isByteBuffer(o)) {
      return;
    }
    DataType oType = dataTypeOf(o);

    if (oType != dtype) {
      // INT8 and UINT8 have the same string name, "byte"
      if (oType.toStringName().equals(dtype.toStringName())) {
        return;
      }

      throw new IllegalArgumentException(
          String.format(
              "Cannot convert between a TensorFlowLite tensor with type %s and a Java "
                  + "object of type %s (which is compatible with the TensorFlowLite type %s).",
              dtype, o.getClass().getName(), oType));
    }
  }

  private void throwIfSrcShapeIsIncompatible(Object src) {
    if (isBuffer(src)) {
      Buffer srcBuffer = (Buffer) src;
      int bytes = numBytes();
      // Note that we allow the client to provide a ByteBuffer even for non-byte Tensors.
      // In such cases, we only care that the raw byte capacity matches the tensor byte capacity.
      int srcBytes =
          isByteBuffer(src) ? srcBuffer.capacity() : srcBuffer.capacity() * dtype.byteSize();
      if (bytes != srcBytes) {
        throw new IllegalArgumentException(
            String.format(
                "Cannot copy to a TensorFlowLite tensor (%s) with %d bytes from a "
                    + "Java Buffer with %d bytes.",
                name(), bytes, srcBytes));
      }
      return;
    }
    int[] srcShape = computeShapeOf(src);
    if (!Arrays.equals(srcShape, shapeCopy)) {
      throw new IllegalArgumentException(
          String.format(
              "Cannot copy to a TensorFlowLite tensor (%s) with shape %s from a Java object "
                  + "with shape %s.",
              name(), Arrays.toString(shapeCopy), Arrays.toString(srcShape)));
    }
  }

  private void throwIfDstShapeIsIncompatible(Object dst) {
    if (isBuffer(dst)) {
      Buffer dstBuffer = (Buffer) dst;
      int bytes = numBytes();
      // Note that we allow the client to provide a ByteBuffer even for non-byte Tensors.
      // In such cases, we only care that the raw byte capacity fits the tensor byte capacity.
      // This is subtly different than Buffer *inputs*, where the size should be exact.
      int dstBytes =
          isByteBuffer(dst) ? dstBuffer.capacity() : dstBuffer.capacity() * dtype.byteSize();
      if (bytes > dstBytes) {
        throw new IllegalArgumentException(
            String.format(
                "Cannot copy from a TensorFlowLite tensor (%s) with %d bytes to a "
                    + "Java Buffer with %d bytes.",
                name(), bytes, dstBytes));
      }
      return;
    }
    int[] dstShape = computeShapeOf(dst);
    if (!Arrays.equals(dstShape, shapeCopy)) {
      throw new IllegalArgumentException(
          String.format(
              "Cannot copy from a TensorFlowLite tensor (%s) with shape %s to a Java object "
                  + "with shape %s.",
              name(), Arrays.toString(shapeCopy), Arrays.toString(dstShape)));
    }
  }

  private static boolean isBuffer(Object o) {
    return o instanceof Buffer;
  }

  private static boolean isByteBuffer(Object o) {
    return o instanceof ByteBuffer;
  }

  private long nativeHandle;
  private final DataType dtype;
  private int[] shapeCopy;
  private final int[] shapeSignatureCopy;
  private final QuantizationParams quantizationParamsCopy;

  private Tensor(long nativeHandle) {
    this.nativeHandle = nativeHandle;
    this.dtype = DataType.fromC(dtype(nativeHandle));
    this.shapeCopy = shape(nativeHandle);
    this.shapeSignatureCopy = shapeSignature(nativeHandle);
    this.quantizationParamsCopy =
        new QuantizationParams(
            quantizationScale(nativeHandle), quantizationZeroPoint(nativeHandle));
  }

  private ByteBuffer buffer() {
    return buffer(nativeHandle).order(ByteOrder.nativeOrder());
  }

  private static native long create(long interpreterHandle, int tensorIndex);

  private static native void delete(long handle);

  private static native ByteBuffer buffer(long handle);

  private static native void writeDirectBuffer(long handle, Buffer src);

  private static native int dtype(long handle);

  private static native int[] shape(long handle);

  private static native int[] shapeSignature(long handle);

  private static native int numBytes(long handle);

  private static native boolean hasDelegateBufferHandle(long handle);

  private static native void readMultiDimensionalArray(long handle, Object dst);

  private static native void writeMultiDimensionalArray(long handle, Object src);

  private static native void writeScalar(long handle, Object src);

  private static native int index(long handle);

  private static native String name(long handle);

  private static native float quantizationScale(long handle);

  private static native int quantizationZeroPoint(long handle);
}
