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

/**
 * A symbolic handle to a tensor produced by an {@link Operation}.
 *
 * <p>An Output is a symbolic handle to a tensor. The value of the Tensor is computed by executing
 * the {@link Operation} in a {@link Session}.
 */
public final class Output {

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

  /**
   * Returns the (possibly partially known) shape of the operation that will produce the tensor
   * referred to by this Output.
   */
  public Shape shape() {
    return new Shape(operation.shape(index));
  }

  private final Operation operation;
  private final int index;
}
