/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow;

import java.util.Objects;

/**
 * A symbolic handle to a tensor produced by an {@link Operation}.
 *
 * <p>An {@code Output<T>} is a symbolic handle to a {@code Tensor<T>}. The value of the tensor is
 * computed by executing the {@link Operation} in a {@link Session}.
 *
 * <p>By implementing the {@link Operand} interface, instances of this class also act as operands to
 * {@link org.tensorflow.op.Op Op} instances.
 */
public final class Output<T> implements Operand<T> {

  /** Handle to the idx-th output of the Operation {@code op}. */
  public Output(Operation op, int idx) {
    operation = op;
    index = idx;
  }

  /** Returns the Operation that will produce the tensor referred to by this Output. */
  public Operation op() {
    return operation;
  }

  /** Returns the index into the outputs of the Operation. */
  public int index() {
    return index;
  }

  /** Returns the (possibly partially known) shape of the tensor referred to by this Output. */
  public Shape shape() {
    return new Shape(operation.shape(index));
  }

  /** Returns the DataType of the tensor referred to by this Output. */
  public DataType dataType() {
    return operation.dtype(index);
  }

  @Override
  public Output<T> asOutput() {
    return this;
  }

  @Override
  public int hashCode() {
    return Objects.hash(operation, index);
  }

  @Override
  public boolean equals(Object o) {
    if (o == this) {
      return true;
    }
    if (o instanceof Output<?>) {
      Output<?> that = (Output<?>) o;
      return index == that.index && operation.equals(that.operation);
    }
    return false;
  }

  @Override
  public String toString() {
    return String.format(
        "<%s '%s:%d' shape=%s dtype=%s>",
        operation.type(), operation.name(), index, shape().toString(), dataType());
  }

  private final Operation operation;
  private final int index;
}
