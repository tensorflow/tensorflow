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

import org.tensorflow.Graph;

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
   * @param graph The graph instance to be managed by the scope.
   */
  public Scope(Graph graph) {
    this(graph, new NameScope());
  }

  /** Returns the graph managed by this scope. */
  public Graph graph() {
    return graph;
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
    return new Scope(graph, nameScope.withSubScope(childScopeName));
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
    return new Scope(graph, nameScope.withName(opName));
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
   * scope.graph().opBuilder("Const", scope.makeOpName("Const"))...
   * }</pre>
   *
   * <p><b>Note:</b> if you provide a composite operator building class (i.e, a class that adds a
   * set of related operations to the graph by calling other operator building code), the provided name 
   * will act as a subscope to all underlying operators.
   *
   * @param defaultName name for the underlying operator.
   * @return unique name for the operator.
   * @throws IllegalArgumentException if the default name is invalid.
   */
  public String makeOpName(String defaultName) {
    return nameScope.makeOpName(defaultName);
  }

  private Scope(Graph graph, NameScope nameScope) {
    this.graph = graph;
    this.nameScope = nameScope;
  }

  private final Graph graph;
  private final NameScope nameScope;
}
