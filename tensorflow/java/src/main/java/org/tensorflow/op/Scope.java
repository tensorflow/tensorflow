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

import org.tensorflow.ExecutionEnvironment;
import org.tensorflow.Operand;
import org.tensorflow.OperationBuilder;

import java.util.ArrayList;

/**
 * Manages groups of related properties when creating Tensorflow Operations, such as a common name
 * prefix.
 *
 * <p>A {@code Scope} is a container for common properties applied to TensorFlow Ops. Normal user
 * code initializes a {@code Scope} and provides it to Operation building classes. For example:
 *
 * <pre>{@code
 * Scope scope = new Scope(graph);
 * Constant c = Constant.create(scope, 42);
 * }</pre>
 *
 * <p>An Operation building class acquires a Scope, and uses it to set properties on the underlying
 * Tensorflow ops. For example:
 *
 * <pre>{@code
 * // An operator class that adds a constant.
 * public class Constant {
 *   public static Constant create(Scope scope, ...) {
 *      scope.graph().opBuilder(
 *        "Const", scope.makeOpName("Const"))
 *        .setAttr(...)
 *        .build()
 *      ...
 *   }
 * }
 * }</pre>
 *
 * <p><b>Scope hierarchy:</b>
 *
 * <p>A {@code Scope} provides various {@code with()} methods that create a new scope. The new scope
 * typically has one property changed while other properties are inherited from the parent scope.
 *
 * <p>An example using {@code Constant} implemented as before:
 *
 * <pre>{@code
 * Scope root = new Scope(graph);
 *
 * // The linear subscope will generate names like linear/...
 * Scope linear = Scope.withSubScope("linear");
 *
 * // This op name will be "linear/W"
 * Constant.create(linear.withName("W"), ...);
 *
 * // This op will be "linear/Const", using the default
 * // name provided by Constant
 * Constant.create(linear, ...);
 *
 * // This op will be "linear/Const_1", using the default
 * // name provided by Constant and making it unique within
 * // this scope
 * Constant.create(linear, ...);
 * }</pre>
 *
 * <p>Scope objects are <b>not</b> thread-safe.
 */
public final class Scope {

  /**
   * Create a new top-level scope.
   *
   * @param env The execution environment used by the scope.
   */
  public Scope(ExecutionEnvironment env) {
    this(env, new NameScope(), new ArrayList<Operand<?>>());
  }

  /** Returns the execution environment used by this scope. */
  public ExecutionEnvironment env() {
    return env;
  }

  /**
   * Returns a new scope where added operations will have the provided name prefix.
   *
   * <p>Ops created with this scope will have {@code name/childScopeName/} as the prefix. The actual
   * name will be unique in the returned scope. All other properties are inherited from the current
   * scope.
   *
   * <p>The child scope name must match the regular expression {@code [A-Za-z0-9.][A-Za-z0-9_.\-]*}
   *
   * @param childScopeName name for the new child scope
   * @return a new subscope
   * @throws IllegalArgumentException if the name is invalid
   */
  public Scope withSubScope(String childScopeName) {
    return new Scope(env, nameScope.withSubScope(childScopeName), controlDependencies);
  }

  /**
   * Return a new scope that uses the provided name for an op.
   *
   * <p>Operations created within this scope will have a name of the form {@code
   * name/opName[_suffix]}. This lets you name a specific operator more meaningfully.
   *
   * <p>Names must match the regular expression {@code [A-Za-z0-9.][A-Za-z0-9_.\-]*}
   *
   * @param opName name for an operator in the returned scope
   * @return a new Scope that uses opName for operations.
   * @throws IllegalArgumentException if the name is invalid
   */
  public Scope withName(String opName) {
    return new Scope(env, nameScope.withName(opName), controlDependencies);
  }

  /**
   * Create a unique name for an operator, using a provided default if necessary.
   *
   * <p>This is normally called only by operator building classes.
   *
   * <p>This method generates a unique name, appropriate for the name scope controlled by this
   * instance. Typical operator building code might look like
   *
   * <pre>{@code
   * scope.env().opBuilder("Const", scope.makeOpName("Const"))...
   * }</pre>
   *
   * <p><b>Note:</b> if you provide a composite operator building class (i.e, a class that creates a
   * set of related operations by calling other operator building code), the provided name will act
   * as a subscope to all underlying operators.
   *
   * @param defaultName name for the underlying operator.
   * @return unique name for the operator.
   * @throws IllegalArgumentException if the default name is invalid.
   */
  public String makeOpName(String defaultName) {
    return nameScope.makeOpName(defaultName);
  }

  private Scope(
      ExecutionEnvironment env, NameScope nameScope, Iterable<Operand<?>> controlDependencies) {
    this.env = env;
    this.nameScope = nameScope;
    this.controlDependencies = controlDependencies;
  }

  /**
   * Returns a new scope where added operations will have the provided control dependencies.
   *
   * <p>Ops created with this scope will have a control edge from each of the provided controls. All
   * other properties are inherited from the current scope.
   *
   * @param controls control dependencies for ops created with the returned scope
   * @return a new scope with the provided control dependencies
   */
  public Scope withControlDependencies(Iterable<Operand<?>> controls) {
    return new Scope(env, nameScope, controls);
  }

  /**
   * Adds each Operand in controlDependencies as a control input to the provided builder.
   *
   * @param builder OperationBuilder to add control inputs to
   */
  public OperationBuilder applyControlDependencies(OperationBuilder builder) {
    for (Operand<?> control : controlDependencies) {
      builder = builder.addControlInput(control.asOutput().op());
    }
    return builder;
  }

  private final ExecutionEnvironment env;
  private final Iterable<Operand<?>> controlDependencies;
  private final NameScope nameScope;
}
