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
 * Performs computation on Tensors.
 *
 * <p>An Operation takes zero or more {@link Tensor}s (produced by other Operations) as input, and 
 * produces zero or more {@link Tensor}s as output.
 */
public interface Operation {

  /** Returns the full name of the Operation. */
  String name();

  /**
   * Returns the type of the operation, i.e., the name of the computation performed by the
   * operation.
   */
  String type();

  /** Returns the number of tensors produced by this operation. */
  int numOutputs();

  /**
   * Returns the size of the list of Tensors produced by this operation.
   *
   * <p>An Operation has multiple named outputs, each of which produces either a single tensor or a
   * list of tensors. This method returns the size of the list of tensors for a specific named
   * output of the operation.
   *
   * @param name identifier of the list of tensors (of which there may be many) produced by this
   *     operation.
   * @return the size of the list of Tensors produced by this named output.
   * @throws IllegalArgumentException if this operation has no output with the provided name.
   */
  int outputListLength(final String name);

  /**
   * Returns symbolic handles to a list of tensors produced by this operation.
   *
   * @param idx index of the first tensor of the list
   * @param length number of tensors in the list
   * @return array of {@code Output}
   */
  Output<?>[] outputList(int idx, int length);

  /**
   * Returns a symbolic handle to one of the tensors produced by this operation.
   *
   * <p>Warning: Does not check that the type of the tensor matches T. It is recommended to call
   * this method with an explicit type parameter rather than letting it be inferred, e.g. {@code
   * operation.<Integer>output(0)}
   *
   * @param <T> The expected element type of the tensors produced by this output.
   * @param idx The index of the output among the outputs produced by this operation.
   */
  <T> Output<T> output(int idx);

  /**
   * Returns the size of the given inputs list of Tensors for this operation.
   *
   * <p>An Operation has multiple named inputs, each of which contains either a single tensor or a
   * list of tensors. This method returns the size of the list of tensors for a specific named input
   * of the operation.
   *
   * @param name identifier of the list of tensors (of which there may be many) inputs to this
   *     operation.
   * @return the size of the list of Tensors produced by this named input.
   * @throws IllegalArgumentException if this operation has no input with the provided name.
   */
  int inputListLength(final String name);
}
