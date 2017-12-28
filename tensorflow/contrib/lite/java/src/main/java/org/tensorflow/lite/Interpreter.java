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

import java.io.File;
import java.nio.MappedByteBuffer;
import java.util.HashMap;
import java.util.Map;
import javax.validation.constraints.NotNull;

/**
 * Driver class to drive model inference with TensorFlow Lite.
 *
 * <p>A {@code Interpreter} encapsulates a pre-trained TensorFlow Lite model, in which operations
 * are executed for model inference.
 *
 * <p>For example, if a model takes only one input and returns only one output:
 *
 * <pre>{@code
 * try (Interpreter interpreter = new Interpreter(file_of_a_tensorflowlite_model)) {
 *   interpreter.run(input, output);
 * }
 * }</pre>
 *
 * <p>If a model takes multiple inputs or outputs:
 *
 * <pre>{@code
 * Object[] inputs = {input0, input1, ...};
 * Map<Integer, Object> map_of_indices_to_outputs = new HashMap<>();
 * float[][][] ith_output = new float[3][2][4];
 * map_of_indices_to_outputs.put(i, ith_output);
 * try (Interpreter interpreter = new Interpreter(file_of_a_tensorflowlite_model)) {
 *   interpreter.runForMultipleInputsOutputs(inputs, map_of_indices_to_outputs);
 * }
 * }</pre>
 *
 * <p>Orders of inputs and outputs are determined when converting TensorFlow model to TensorFlowLite
 * model with Toco.
 *
 * <p><b>WARNING:</b>Instances of a {@code Interpreter} is <b>not</b> thread-safe. A {@code
 * Interpreter} owns resources that <b>must</b> be explicitly freed by invoking {@link #close()}
 */
public final class Interpreter implements AutoCloseable {

  /**
   * Initializes a {@code Interpreter}
   *
   * @param modelFile: a File of a pre-trained TF Lite model.
   */
  public Interpreter(@NotNull File modelFile) {
    if (modelFile == null) {
      return;
    }
    wrapper = new NativeInterpreterWrapper(modelFile.getAbsolutePath());
  }

  /**
   * Initializes a {@code Interpreter} with a {@code MappedByteBuffer} to the model file.
   *
   * <p>The {@code MappedByteBuffer} should remain unchanged after the construction of a {@code
   * Interpreter}.
   */
  public Interpreter(@NotNull MappedByteBuffer mappedByteBuffer) {
    wrapper = new NativeInterpreterWrapper(mappedByteBuffer);
  }

  /**
   * Runs model inference if the model takes only one input, and provides only one output.
   *
   * @param input an array or multidimensional array, or a {@link ByteBuffer} of primitive types
   *     including int, float, long, and byte. {@link ByteBuffer} is the preferred way to pass large
   *     input data. When {@link ByteBuffer} is used, its content should remain unchanged until
   *     model inference is done.
   * @param output a multidimensional array of output data.
   */
  public void run(@NotNull Object input, @NotNull Object output) {
    Object[] inputs = {input};
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, output);
    runForMultipleInputsOutputs(inputs, outputs);
  }

  /**
   * Runs model inference if the model takes multiple inputs, or returns multiple outputs.
   *
   * @param inputs an array of input data. The inputs should be in the same order as inputs of the
   *     model. Each input can be an array or multidimensional array, or a {@link ByteBuffer} of
   *     primitive types including int, float, long, and byte. {@link ByteBuffer} is the preferred
   *     way to pass large input data. When {@link ByteBuffer} is used, its content should remain
   *     unchanged until model inference is done.
   * @param outputs a map mapping output indices to multidimensional arrays of output data. It only
   *     needs to keep entries for the outputs to be used.
   */
  public void runForMultipleInputsOutputs(
      @NotNull Object[] inputs, @NotNull Map<Integer, Object> outputs) {
    if (wrapper == null) {
      throw new IllegalStateException("The Interpreter has already been closed.");
    }
    Tensor[] tensors = wrapper.run(inputs);
    if (outputs == null || tensors == null || outputs.size() > tensors.length) {
      throw new IllegalArgumentException("Outputs do not match with model outputs.");
    }
    final int size = tensors.length;
    for (Integer idx : outputs.keySet()) {
      if (idx == null || idx < 0 || idx >= size) {
        throw new IllegalArgumentException(
            String.format("Invalid index of output %d (should be in range [0, %d))", idx, size));
      }
      tensors[idx].copyTo(outputs.get(idx));
    }
  }

  /**
   * Resizes idx-th input of the native model to the given dims.
   *
   * <p>IllegalArgumentException will be thrown if it fails to resize.
   */
  public void resizeInput(int idx, @NotNull int[] dims) {
    if (wrapper == null) {
      throw new IllegalStateException("The Interpreter has already been closed.");
    }
    wrapper.resizeInput(idx, dims);
  }

  /**
   * Gets index of an input given the op name of the input.
   *
   * <p>IllegalArgumentException will be thrown if the op name does not exist in the model file used
   * to initialize the {@link Interpreter}.
   */
  public int getInputIndex(String opName) {
    if (wrapper == null) {
      throw new IllegalStateException("The Interpreter has already been closed.");
    }
    return wrapper.getInputIndex(opName);
  }

  /**
   * Gets index of an output given the op name of the output.
   *
   * <p>IllegalArgumentException will be thrown if the op name does not exist in the model file used
   * to initialize the {@link Interpreter}.
   */
  public int getOutputIndex(String opName) {
    if (wrapper == null) {
      throw new IllegalStateException("The Interpreter has already been closed.");
    }
    return wrapper.getOutputIndex(opName);
  }

  /** Release resources associated with the {@code Interpreter}. */
  @Override
  public void close() {
    wrapper.close();
    wrapper = null;
  }

  NativeInterpreterWrapper wrapper;
}
