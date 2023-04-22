/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
 * Base class for {@link Operation} implementations.
 *
 * <p>As opposed to {@link Operation} itself, this class is package private and therefore its usage
 * is limited to internal purposes only.
 */
abstract class AbstractOperation implements Operation {

  @Override
  public Output<?>[] outputList(int idx, int length) {
    Output<?>[] outputs = new Output<?>[length];
    for (int i = 0; i < length; ++i) {
      outputs[i] = output(idx + i);
    }
    return outputs;
  }

  @Override
  @SuppressWarnings({"rawtypes", "unchecked"})
  public <T> Output<T> output(int idx) {
    return new Output(this, idx);
  }

  @Override
  public String toString() {
    return String.format("<%s '%s'>", type(), name());
  }

  /**
   * Returns the native handle of the {@code outputIdx}th output of this operation.
   *
   * <p>The nature of the returned value varies depending on current the execution environment.
   *
   * <ul>
   *   <li>In eager mode, the value is a handle to the tensor returned at this output.
   *   <li>In graph mode, the value is a handle to the operation itself, which should be paired with
   *       the index of the output when calling the native layer.
   * </ul>
   *
   * @param outputIdx index of the output in this operation
   * @return a native handle, see method description for more details
   */
  abstract long getUnsafeNativeHandle(int outputIdx);

  /**
   * Returns the shape of the tensor of the {@code outputIdx}th output of this operation.
   *
   * @param outputIdx index of the output of this operation
   * @return output tensor shape
   */
  abstract long[] shape(int outputIdx);

  /**
   * Returns the datatype of the tensor of the {@code outputIdx}th output of this operation.
   *
   * @param outputIdx index of the output of this operation
   * @return output tensor datatype
   */
  abstract DataType dtype(int outputIdx);

  /**
   * Returns the tensor of the {@code outputIdx}th output of this operation.
   *
   * <p>This is only supported in an eager execution environment.
   *
   * @param outputIdx index of the output of this operation
   * @return output tensor
   */
  abstract Tensor<?> tensor(int outputIdx);
}
