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

package org.tensorflow;

/**
 * A tensor that can be used as an operand for operation wrappers.
 *
 * <p>The {@code Input} interface is at the base of the Java Ops API by allowing any object that
 * wraps an input tensor to be passed as an operand to an operation.
 *
 * <pre>{@code
 * // Output implements Input.
 * Output image = ops.image().decodeJpeg(...).image();
 * ops.math().cast(image, DataType.FLOAT);
 *
 * // DecodeJpeg operation also implements Input, returning the image as its output.
 * ops.math().cast(ops.image().decodeJpeg(...), DataType.FLOAT);
 * }</pre>
 *
 * <p>Additionally, objects implementing {@code Iterable<Input>} can be passed in input to
 * operations handling a list of tensors as one of its operand.
 *
 * <pre>{@code
 * // List<Output> extends from Iterable<Input>
 * List<Output> outputs = ops.array().split(...).output();
 * ops.array().concat(0, outputs);
 *
 * // Split implements Iterable<Input>, returning an iterator on the split tensors.
 * ops.arrays().concat(0, ops.array().split(...));
 * }</pre>
 *
 * <p>The handle of the tensor to be passed in input is encapsulated by an instance of {@code
 * Output}, which could be retrieved by calling {@link #asOutput()}. Therefore, the {@code Output}
 * class and the {@code Input} interface are complementary to represent a tensor being passed
 * between two nodes of the graph.
 */
public interface Input {

  /**
   * Returns the symbolic handle of a tensor.
   *
   * <p>This method is called by the Java Ops API to retrieve the symbolic handle of a tensor added
   * in input to an operation.
   *
   * @see {@link OperationBuilder#addInput(Output)}.
   */
  Output asOutput();
}
