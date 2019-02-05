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

package org.tensorflow.op.annotation;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Annotation used by classes to make TensorFlow operations conveniently accessible via {@code
 * org.tensorflow.op.Ops}.
 *
 * <p>An annotation processor ({@code org.tensorflow.processor.OperatorProcessor}) builds the
 * {@code Ops} class by aggregating all classes annotated as {@code @Operator}s. Each annotated
 * class <b>must</b> have at least one public static factory method named {@code create} that
 * accepts a {@link org.tensorflow.op.Scope} as its first argument. The processor then adds a
 * convenience method in the {@code Ops} class. For example:
 *
 * <pre>{@code
 * @Operator
 * public final class MyOp implements Op {
 *   public static MyOp create(Scope scope, Operand operand) {
 *     ...
 *   }
 * }
 * }</pre>
 *
 * <p>results in a method in the {@code Ops} class
 *
 * <pre>{@code
 * import org.tensorflow.op.Ops;
 * ...
 * Ops ops = Ops.create(graph);
 * ...
 * ops.myOp(operand);
 * // and has exactly the same effect as calling
 * // MyOp.create(ops.getScope(), operand);
 * }</pre>
 */
@Documented
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.SOURCE)
public @interface Operator {
  /**
   * Specify an optional group within the {@code Ops} class.
   *
   * <p>By default, an annotation processor will create convenience methods directly in the {@code
   * Ops} class. An annotated operator may optionally choose to place the method within a group. For
   * example:
   *
   * <pre>{@code
   * @Operator(group="math")
   * public final class Add extends PrimitiveOp implements Operand {
   *   ...
   * }
   * }</pre>
   *
   * <p>results in the {@code add} method placed within a {@code math} group within the {@code Ops}
   * class.
   *
   * <pre>{@code
   * ops.math().add(...);
   * }</pre>
   *
   * <p>The group name must be a <a
   * href="https://docs.oracle.com/javase/specs/jls/se7/html/jls-3.html#jls-3.8">valid Java
   * identifier</a>.
   */
  String group() default "";

  /**
   * Name for the wrapper method used in the {@code Ops} class.
   *
   * <p>By default, a processor derives the method name in the {@code Ops} class from the class name
   * of the operator. This attribute allow you to provide a different name instead. For example:
   *
   * <pre>{@code
   * @Operator(name="myOperation")
   * public final class MyRealOperation implements Operand {
   *   public static MyRealOperation create(...)
   * }
   * }</pre>
   *
   * <p>results in this method added to the {@code Ops} class
   *
   * <pre>{@code
   * ops.myOperation(...);
   * // and is the same as calling
   * // MyRealOperation.create(...)
   * }</pre>
   *
   * <p>The name must be a <a
   * href="https://docs.oracle.com/javase/specs/jls/se7/html/jls-3.html#jls-3.8">valid Java
   * identifier</a>.
   */
  String name() default "";
}
