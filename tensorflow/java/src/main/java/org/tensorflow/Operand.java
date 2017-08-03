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
 * Interface implemented by operands of a TensorFlow operation.
 *
 * <p>Example usage:
 *
 * <pre>{@code
 * // The "decodeJpeg" operation can be used as an operand to the "cast" operation
 * Operand decodeJpeg = ops.image().decodeJpeg(...);
 * ops.math().cast(decodeJpeg, DataType.FLOAT);
 *
 * // The output "y" of the "unique" operation can be used as an operand to the "cast" operation
 * Output y = ops.array().unique(...).y();
 * ops.math().cast(y, DataType.FLOAT);
 *
 * // The "split" operation can be used as operand list to the "concat" operation
 * Iterable<? extends Operand> split = ops.array().split(...);
 * ops.array().concat(0, split);
 * }</pre>
 */
public interface Operand {

  /**
   * Returns the symbolic handle of a tensor.
   *
   * <p>Inputs to TensorFlow operations are outputs of another TensorFlow operation. This method is
   * used to obtain a symbolic handle that represents the computation of the input.
   *
   * @see OperationBuilder#addInput(Output)
   */
  Output asOutput();
}
