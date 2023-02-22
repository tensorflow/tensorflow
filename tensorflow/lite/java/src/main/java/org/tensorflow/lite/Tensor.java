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

/**
 * A typed multi-dimensional array used in Tensorflow Lite.
 *
 * <p>The native handle of a {@code Tensor} is managed by {@code NativeInterpreterWrapper}, and does
 * not needed to be closed by the client. However, once the {@code NativeInterpreterWrapper} has
 * been closed, the tensor handle will be invalidated.
 */
public interface Tensor {

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
  class QuantizationParams {
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

  /** Returns the {@link DataType} of elements stored in the Tensor. */
  DataType dataType();

  /**
   * Returns the number of dimensions (sometimes referred to as <a
   * href="https://www.tensorflow.org/resources/dims_types.html#rank">rank</a>) of the Tensor.
   *
   * <p>Will be 0 for a scalar, 1 for a vector, 2 for a matrix, 3 for a 3-dimensional tensor etc.
   */
  int numDimensions();

  /** Returns the size, in bytes, of the tensor data. */
  int numBytes();

  /** Returns the number of elements in a flattened (1-D) view of the tensor. */
  int numElements();

  /**
   * Returns the <a href="https://www.tensorflow.org/resources/dims_types.html#shape">shape</a> of
   * the Tensor, i.e., the sizes of each dimension.
   *
   * @return an array where the i-th element is the size of the i-th dimension of the tensor.
   */
  int[] shape();

  /**
   * Returns the original <a
   * href="https://www.tensorflow.org/resources/dims_types.html#shape">shape</a> of the Tensor,
   * i.e., the sizes of each dimension - before any resizing was performed. Unknown dimensions are
   * designated with a value of -1.
   *
   * @return an array where the i-th element is the size of the i-th dimension of the tensor.
   */
  int[] shapeSignature();

  /**
   * Returns the (global) index of the tensor within the subgraph of the owning interpreter.
   *
   * @hide
   */
  int index();

  /**
   * Returns the name of the tensor within the owning interpreter.
   *
   * @hide
   */
  String name();

  /**
   * Returns the quantization parameters of the tensor within the owning interpreter.
   *
   * <p>Only quantized tensors have valid {@code QuantizationParameters}. For tensor that are not
   * quantized, the values of scale and zero_point are both 0.
   */
  QuantizationParams quantizationParams();

  /**
   * Returns a read-only {@code ByteBuffer} view of the tensor data.
   *
   * <p>In general, this method is most useful for obtaining a read-only view of output tensor data,
   * *after* inference has been executed (e.g., via {@link InterpreterApi#run(Object,Object)}). In
   * particular, some graphs have dynamically shaped outputs, which can make feeding a predefined
   * output buffer to the interpreter awkward. Example usage:
   *
   * <pre> {@code
   * interpreter.run(input, null);
   * ByteBuffer outputBuffer = interpreter.getOutputTensor(0).asReadOnlyBuffer();
   * // Copy or read from outputBuffer.}</pre>
   *
   * <p>WARNING: If the tensor has not yet been allocated, e.g., before inference has been executed,
   * the result is undefined. Note that the underlying tensor pointer may also change when the
   * tensor is invalidated in any way (e.g., if inference is executed, or the graph is resized), so
   * it is *not* safe to hold a reference to the returned buffer beyond immediate use directly
   * following inference. Example *bad* usage:
   *
   * <pre> {@code
   * ByteBuffer outputBuffer = interpreter.getOutputTensor(0).asReadOnlyBuffer();
   * interpreter.run(input, null);
   * // Copy or read from outputBuffer (which may now be invalid).}</pre>
   *
   * @throws IllegalArgumentException if the tensor data has not been allocated.
   */
  ByteBuffer asReadOnlyBuffer();
}
