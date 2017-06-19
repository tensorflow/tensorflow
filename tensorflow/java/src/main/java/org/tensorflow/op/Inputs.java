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

package org.tensorflow.op;

import java.util.ArrayList;
import java.util.List;

import org.tensorflow.Input;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.VariableInput;

/** Utilities for manipulating input related types and lists. */
public final class Inputs {

  /**
   * Converts a list of {@link Input} into an array of {@link Output}.
   *
   * <p>Some wrappers need to convert back a list of inputs into an array of outputs in order to
   * build an operation, see {@link OperationBuilder#addInputList(Output[])}.
   *
   * @param inputs an iteration of input operands
   * @return an array of outputs
   */
  public static Output[] asOutputs(Iterable<? extends Input> inputs) {
    List<Output> outputList = new ArrayList<>();
    for (Input input : inputs) {
      outputList.add(input.asOutput());
    }
    return outputList.toArray(new Output[outputList.size()]);
  }

  /**
   * Converts an {@link Output} into a {@link VariableInput}.
   *
   * @param output the output of an operation
   * @return an input that could be passed as a variable operand
   */
  public static VariableInput asVariable(final Output output) {
    return new VariableInput() {

      @Override
      public Output asOutput() {
        return output;
      }
    };
  }

  // Disabled constructor
  private Inputs() {}
}
