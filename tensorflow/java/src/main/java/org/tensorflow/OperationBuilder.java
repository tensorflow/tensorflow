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
 * A builder for {@link Operation}s.
 *
 * <p>For example, the following uses the builder to create an operation that produces the constant
 * "3" as its output:
 *
 * <pre>{@code
 * // env is an ExecutionEnvironment, such as a Graph instance.
 * try (Tensor c1 = Tensor.create(3.0f)) {
 *   env.opBuilder("Const", "MyConst")
 *       .setAttr("dtype", c1.dataType())
 *       .setAttr("value", c1)
 *       .build();
 * }
 * }</pre>
 */
public interface OperationBuilder {

  /**
   * Build the {@link Operation}.
   *
   * <p>The following action will also be performed depending on the current execution environment.
   *
   * <ul>
   *   <li>In eager mode, the result of the operation will be computed immediately.
   *   <li>In graph mode, the operation will be added as a node to the graph to be executed later,
   *       when running a {@link Session}.
   * </ul>
   *
   * <p>The OperationBuilder is not usable after build() returns.
   */
  public Operation build();

  /**
   * Add the output of another operation as the next input of the operation being built.
   *
   * @param input {@link Output} supposed to be the input of the operation being built.
   * @return the OperationBuilder instance for chaining.
   */
  public OperationBuilder addInput(Output<?> input);

  /**
   * Add the outputs of another operation as the next inputs of the operation being built.
   *
   * @param inputs list of {@link Output} supposed to be the inputs of the operation being built.
   * @return the OperationBuilder instance for chaining.
   */
  public OperationBuilder addInputList(Output<?>[] inputs);

  /**
   * Ensure that the operation does not execute before the control operation does.
   *
   * <p>A control input is an Operation that must be executed before running the operation currently
   * being built.
   *
   * <p>For example, an Assert operation may be added as a control input for this operation. The
   * Assert now behaves as a pre-condition that will always verify itself before running the
   * operation.
   *
   * @param control operation that must be executed before running this operation.
   * @return the OperationBuilder instance for chaining.
   */
  public OperationBuilder addControlInput(Operation control);

  /**
   * Sets the device requested for computing the operation being built.
   *
   * @param device the requested device, as a string
   * @return the OperationBuilder instance for chaining.
   */
  public OperationBuilder setDevice(String device);

  /**
   * Sets the string values of an attribute of the operation being built.
   *
   * @param name attribute name
   * @param value attribute values
   * @return the OperationBuilder instance for chaining.
   */
  public OperationBuilder setAttr(String name, String[] value);

  /**
   * Sets the string value of an attribute of the operation being built.
   *
   * @param name attribute name
   * @param value attribute value
   * @return the OperationBuilder instance for chaining.
   */
  public OperationBuilder setAttr(String name, String value);

  /**
   * Sets the byte values of an attribute of the operation being built.
   *
   * @param name attribute name
   * @param value attribute values
   * @return the OperationBuilder instance for chaining.
   */
  public OperationBuilder setAttr(String name, byte[] value);

  /**
   * Sets the long value of an attribute of the operation being built.
   *
   * @param name attribute name
   * @param value attribute value
   * @return the OperationBuilder instance for chaining.
   */
  public OperationBuilder setAttr(String name, long value);

  /**
   * Sets the long values of an attribute of the operation being built.
   *
   * @param name attribute name
   * @param value attribute values
   * @return the OperationBuilder instance for chaining.
   */
  public OperationBuilder setAttr(String name, long[] value);

  /**
   * Sets the float value of an attribute of the operation being built.
   *
   * @param name attribute name
   * @param value attribute value
   * @return the OperationBuilder instance for chaining.
   */
  public OperationBuilder setAttr(String name, float value);

  /**
   * Sets the float values of an attribute of the operation being built.
   *
   * @param name attribute name
   * @param value attribute values
   * @return the OperationBuilder instance for chaining.
   */
  public OperationBuilder setAttr(String name, float[] value);

  /**
   * Sets the boolean value of an attribute of the operation being built.
   *
   * @param name attribute name
   * @param value attribute value
   * @return the OperationBuilder instance for chaining.
   */
  public OperationBuilder setAttr(String name, boolean value);

  /**
   * Sets the boolean values of an attribute of the operation being built.
   *
   * @param name attribute name
   * @param value attribute values
   * @return the OperationBuilder instance for chaining.
   */
  public OperationBuilder setAttr(String name, boolean[] value);

  /**
   * Sets the type value of an attribute of the operation being built.
   *
   * @param name attribute name
   * @param value attribute value
   * @return the OperationBuilder instance for chaining.
   */
  public OperationBuilder setAttr(String name, DataType value);

  /**
   * Sets the type values of an attribute of the operation being built.
   *
   * @param name attribute name
   * @param value attribute values
   * @return the OperationBuilder instance for chaining.
   */
  public OperationBuilder setAttr(String name, DataType[] value);

  /**
   * Sets the tensor value of an attribute of the operation being built.
   *
   * @param name attribute name
   * @param value attribute value
   * @return the OperationBuilder instance for chaining.
   */
  public OperationBuilder setAttr(String name, Tensor<?> value);

  /**
   * Sets the tensor values of an attribute of the operation being built.
   *
   * @param name attribute name
   * @param value attribute values
   * @return the OperationBuilder instance for chaining.
   */
  public OperationBuilder setAttr(String name, Tensor<?>[] value);

  /**
   * Sets the shape value of an attribute of the operation being built.
   *
   * @param name attribute name
   * @param value attribute value
   * @return the OperationBuilder instance for chaining.
   */
  public OperationBuilder setAttr(String name, Shape value);

  /**
   * Sets the shape values of an attribute of the operation being built.
   *
   * @param name attribute name
   * @param value attribute values
   * @return the OperationBuilder instance for chaining.
   */
  public OperationBuilder setAttr(String name, Shape[] value);
}
