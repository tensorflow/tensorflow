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

package org.tensorflow.op.core;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.Output;
import org.tensorflow.Tensor;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;
import org.tensorflow.op.annotation.Operator;

/** An operator producing a constant value. */
@Operator
public final class Constant extends PrimitiveOp implements Operand {
  /**
   * Create a constant from a Java object.
   *
   * <p>The argument {@code object} is first converted into a Tensor using {@link
   * org.tensorflow.Tensor#create(Object)}, so only Objects supported by this method must be
   * provided. For example:
   *
   * <pre>{@code
   * Constant.create(scope, 7); // returns a constant scalar tensor 7
   * }</pre>
   *
   * @param scope is a scope used to add the underlying operation.
   * @param object a Java object representing the constant.
   * @see org.tensorflow.Tensor#create(Object) Tensor.create
   */
  public static Constant create(Scope scope, Object object) {
    try (Tensor value = Tensor.create(object)) {
      return createWithTensor(scope, value);
    }
  }

  /**
   * Create a {@link DataType#INT32} constant with data from the given buffer.
   *
   * <p>Creates a constant with the given shape by copying elements from the buffer (starting from
   * its current position) into the tensor. For example, if {@code shape = {2,3} } (which represents
   * a 2x3 matrix) then the buffer must have 6 elements remaining, which will be consumed by this
   * method.
   *
   * @param scope is a scope used to add the underlying operation.
   * @param shape the tensor shape.
   * @param data a buffer containing the tensor data.
   * @throws IllegalArgumentException If the tensor shape is not compatible with the buffer
   */
  public static Constant create(Scope scope, long[] shape, IntBuffer data) {
    try (Tensor value = Tensor.create(shape, data)) {
      return createWithTensor(scope, value);
    }
  }

  /**
   * Create a {@link DataType#FLOAT} constant with data from the given buffer.
   *
   * <p>Creates a constant with the given shape by copying elements from the buffer (starting from
   * its current position) into the tensor. For example, if {@code shape = {2,3} } (which represents
   * a 2x3 matrix) then the buffer must have 6 elements remaining, which will be consumed by this
   * method.
   *
   * @param scope is a scope used to add the underlying operation.
   * @param shape the tensor shape.
   * @param data a buffer containing the tensor data.
   * @throws IllegalArgumentException If the tensor shape is not compatible with the buffer
   */
  public static Constant create(Scope scope, long[] shape, FloatBuffer data) {
    try (Tensor value = Tensor.create(shape, data)) {
      return createWithTensor(scope, value);
    }
  }

  /**
   * Create a {@link DataType#DOUBLE} constant with data from the given buffer.
   *
   * <p>Creates a constant with the given shape by copying elements from the buffer (starting from
   * its current position) into the tensor. For example, if {@code shape = {2,3} } (which represents
   * a 2x3 matrix) then the buffer must have 6 elements remaining, which will be consumed by this
   * method.
   *
   * @param scope is a scope used to add the underlying operation.
   * @param shape the tensor shape.
   * @param data a buffer containing the tensor data.
   * @throws IllegalArgumentException If the tensor shape is not compatible with the buffer
   */
  public static Constant create(Scope scope, long[] shape, DoubleBuffer data) {
    try (Tensor value = Tensor.create(shape, data)) {
      return createWithTensor(scope, value);
    }
  }

  /**
   * Create a {@link DataType#INT64} constant with data from the given buffer.
   *
   * <p>Creates a constant with the given shape by copying elements from the buffer (starting from
   * its current position) into the tensor. For example, if {@code shape = {2,3} } (which represents
   * a 2x3 matrix) then the buffer must have 6 elements remaining, which will be consumed by this
   * method.
   *
   * @param scope is a scope used to add the underlying operation.
   * @param shape the tensor shape.
   * @param data a buffer containing the tensor data.
   * @throws IllegalArgumentException If the tensor shape is not compatible with the buffer
   */
  public static Constant create(Scope scope, long[] shape, LongBuffer data) {
    try (Tensor value = Tensor.create(shape, data)) {
      return createWithTensor(scope, value);
    }
  }

  /**
   * Create a constant with data from the given buffer.
   *
   * <p>Creates a Constant with the provided shape of any type where the constant data has been
   * encoded into {@code data} as per the specification of the TensorFlow <a
   * href="https://www.tensorflow.org/code/tensorflow/c/c_api.h">C API</a>.
   *
   * @param scope is a scope used to add the underlying operation.
   * @param dataType the tensor datatype.
   * @param shape the tensor shape.
   * @param data a buffer containing the tensor data.
   * @throws IllegalArgumentException If the tensor datatype or shape is not compatible with the
   *     buffer
   */
  public static Constant create(Scope scope, DataType dataType, long[] shape, ByteBuffer data) {
    try (Tensor value = Tensor.create(dataType, shape, data)) {
      return createWithTensor(scope, value);
    }
  }

  private static Constant createWithTensor(Scope scope, Tensor value) {
    return new Constant(
        scope
            .graph()
            .opBuilder("Const", scope.makeOpName("Const"))
            .setAttr("value", value)
            .setAttr("dtype", value.dataType())
            .build());
  }

  @Override
  public Output asOutput() {
    return output;
  }

  private Constant(Operation operation) {
    super(operation);
    output = operation.output(0);
  }

  private final Output output;
}
