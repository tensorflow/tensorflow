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

/** A helper class for internal tests. */
public class TestHelper {

  /**
   * Gets the last inference duration in nanoseconds. It returns null if there is no previous
   * inference run or the last inference run failed.
   *
   * @param interpreter an instance of {@code Interpreter}. If it is not initialized, an {@code
   *     IllegalArgumentException} will be thrown.
   */
  public static Long getLastNativeInferenceDurationNanoseconds(Interpreter interpreter) {
    if (interpreter != null && interpreter.wrapper != null) {
      return interpreter.wrapper.getLastNativeInferenceDurationNanoseconds();
    } else {
      throw new IllegalArgumentException("Interpreter has not initialized; Failed to get latency.");
    }
  }

  /**
   * Gets the dimensions of an input.
   *
   * @param interpreter an instance of {@code Interpreter}. If it is not initialized, an {@code
   *     IllegalArgumentException} will be thrown.
   * @param index an integer index of the input. If it is invalid, an {@code
   *     IllegalArgumentException} will be thrown.
   */
  public static int[] getInputDims(Interpreter interpreter, int index) {
    if (interpreter != null && interpreter.wrapper != null) {
      return interpreter.wrapper.getInputTensor(index).shape();
    } else {
      throw new IllegalArgumentException(
          "Interpreter has not initialized;" + " Failed to get input dimensions.");
    }
  }

  /**
   * Gets the string name of the data type of an input.
   *
   * @param interpreter an instance of {@code Interpreter}. If it is not initialized, an {@code
   *     IllegalArgumentException} will be thrown.
   * @param index an integer index of the input. If it is invalid, an {@code
   *     IllegalArgumentException} will be thrown.
   * @return string name of the data type. Possible values include "float", "int", "byte", and
   *     "long".
   */
  public static String getInputDataType(Interpreter interpreter, int index) {
    if (interpreter != null && interpreter.wrapper != null) {
      return DataTypeUtils.toStringName(interpreter.wrapper.getInputTensor(index).dataType());
    } else {
      throw new IllegalArgumentException(
          "Interpreter has not initialized;" + " Failed to get input data type.");
    }
  }

  /**
   * Gets the string name of the data type of an output.
   *
   * @param interpreter an instance of {@code Interpreter}. If it is not initialized, an {@code
   *     IllegalArgumentException} will be thrown.
   * @param index an integer index of the output. If it is invalid, an {@code
   *     IllegalArgumentException} will be thrown.
   * @return string name of the data type. Possible values include "float", "int", "byte", and
   *     "long".
   */
  public static String getOutputDataType(Interpreter interpreter, int index) {
    if (interpreter != null && interpreter.wrapper != null) {
      return DataTypeUtils.toStringName(interpreter.wrapper.getOutputTensor(index).dataType());
    } else {
      throw new IllegalArgumentException(
          "Interpreter has not initialized;" + " Failed to get output data type.");
    }
  }
}
