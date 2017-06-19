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
import java.util.Collections;
import java.util.List;

import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;

/**
 * A helper for wrapping a new operation.
 *
 * <p>It provides utilities for wrappers to build a single operation and collect its output(s)
 * afterward.
 *
 * <pre>{@code
 * public final class Unique extends AbstractSingleOp {
 *    ...
 *    public static Unique create(Scope s, Input x) {
 *        OperationHelper unique = OperationHelper.create(s, "Unique");
 *        unique.builder().input(x.asOutput());
 *        return new Unique(unique);
 *    }
 *    ...
 *    private Unique(OperationHelper unique) {
 *        super(unique.operation());
 *        y = unique.nextOutput();
 *        idx = unique.nextOutput();
 *    }
 * }
 * }</pre>
 *
 * <p>{@code OperationHelper} objects are designed for a single usage and should be discarded once
 * the operation wrapper has been created. They are <b>not</b> thread-safe.
 */
public final class OperationHelper {

  /**
   * Creates a helper for a new operation.
   *
   * <p>By default, the name of the operation is its {@code opType} in lowercase, unless a different
   * name has been provided using {@link Scope#withName(String)}.
   *
   * @param scope operation scope
   * @param opType operation type name
   * @return new instance of this class
   */
  public static OperationHelper create(Scope scope, String opType) {
    return new OperationHelper(
        scope.graph().opBuilder(opType, scope.makeOpName(opType.toLowerCase())));
  }

  /**
   * Returns the builder for the operation.
   *
   * <p>Wrappers can add inputs and attributes to the builder to configure the underlying operation.
   * After the operation has been built (i.e. the first time {@link #operation()} is called), any
   * further modifications to the builder are ignored.
   *
   * <p>Additionally, the {@code build()} method of the builder must <b>not</b> be called explicitly
   * outside this class, {@link #operation()} must be used method instead.
   *
   * @return operation builder
   */
  public OperationBuilder builder() {
    return builder;
  }

  /**
   * Retrieve and optionally build the operation.
   *
   * <p>If the operation is retrieved for the first time, it will be built with inputs and
   * attributes collected so far by the operation {@link #builder()}.
   *
   * @return an operation
   * @see {@link #builder()}.
   */
  public Operation operation() {
    if (unsafeOperation == null) {
      unsafeOperation = builder.build();
    }
    return unsafeOperation;
  }

  /**
   * Returns the next output of the operation.
   *
   * <p>All outputs collected with this method or with {@link #nextOutputList(String)} are retrieved
   * sequentially, like an iterator would do.
   *
   * <p>This method invokes {@link #operation()} internally to build the operation if it has not
   * been done yet.
   *
   * @return an output
   */
  public Output nextOutput() {
    return operation().output(outputIndex++);
  }

  /**
   * Returns the next output list of the built operation.
   *
   * <p>All outputs collected with this method or with {@link #nextOutput()} are retrieved
   * sequentially, like an iterator would do.
   *
   * <p>This method invokes {@link #operation()} internally to build the operation if it has not
   * been done yet.
   *
   * @param name name of the list
   * @return an output list
   */
  public List<Output> nextOutputList(String name) {
    int count = operation().outputListLength(name);
    List<Output> outputs = new ArrayList<>(count);
    while (count-- > 0) {
      outputs.add(nextOutput());
    }
    return Collections.unmodifiableList(outputs);
  }

  private final OperationBuilder builder;
  private Operation unsafeOperation;
  private int outputIndex;

  // Private constructor
  private OperationHelper(OperationBuilder builder) {
    this.builder = builder;
    unsafeOperation = null;
    outputIndex = 0;
  }
}
