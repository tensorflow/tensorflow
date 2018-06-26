/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.op.core;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.op.Op;
import org.tensorflow.op.Operands;
import org.tensorflow.op.Scope;
import org.tensorflow.op.annotation.Operator;

/**
 * Adds operations to compute the partial derivatives of sum of {@code y}s w.r.t {@code x}s,
 * i.e., {@code d(y_1 + y_2 + ...)/dx_1, d(y_1 + y_2 + ...)/dx_2...}
 * <p> 
 * If {@code Options.dx()} values are set, they are as the initial symbolic partial derivatives of some loss 
 * function {@code L} w.r.t. {@code y}. {@code Options.dx()} must have the size of {@code y}.
 * <p>
 * If {@code Options.dx()} is not set, the implementation will use dx of {@code OnesLike} for all
 * shapes in {@code y}.
 * <p>
 * The partial derivatives are returned in output {@code dy}, with the size of {@code x}.
 * <p>
 * Example of usage:
 * <pre>{@code
 * Gradients gradients = Gradients.create(scope, Arrays.asList(loss), Arrays.asList(w, b));
 * 
 * Constant<Float> alpha = ops.constant(1.0f, Float.class);
 * ApplyGradientDescent.create(scope, w, alpha, gradients.<Float>dy(0));
 * ApplyGradientDescent.create(scope, b, alpha, gradients.<Float>dy(1));
 * }</pre>
 */
@Operator
public class Gradients implements Op, Iterable<Operand<?>> {

  /**
   * Optional attributes for {@link Gradients}
   */
  public static class Options {
    
    /**
     * @param dx partial derivatives of some loss function {@code L} w.r.t. {@code y}
     * @return this option builder
     */
    public Options dx(Iterable<Operand<?>> dx) {
      this.dx = dx;
      return this;
    }
    
    private Iterable<Operand<?>> dx;
    
    private Options() {
    }
  }

  /**
   * Adds gradients computation ops to the graph according to scope.
   * 
   * @param scope current graph scope
   * @param y outputs of the function to derive
   * @param x inputs of the function for which partial derivatives are computed
   * @param options carries optional attributes values
   * @return a new instance of {@code Gradients}
   */
  public static Gradients create(Scope scope, Iterable<Operand<?>> y, Iterable<Operand<?>> x, Options... options) {
    Output<?>[] dx = null;
    if (options != null) {
      for (Options opts : options) {
        if (opts.dx != null) {
          dx = Operands.asOutputs(opts.dx);
        }
      }
    }
    Output<?>[] dy = scope.graph().addGradients(scope.prefix(), Operands.asOutputs(y), Operands.asOutputs(x), dx);
    return new Gradients(Arrays.asList(dy));
  }

  /**
   * Adds gradients computation ops to the graph according to scope.
   * 
   * This is a simplified version of {@link #create(Scope, Iterable, Iterable, Options...)} where {@code y} is
   * a single output.
   * 
   * @param scope current graph scope
   * @param y output of the function to derive
   * @param x inputs of the function for which partial derivatives are computed
   * @param options carries optional attributes values
   * @return a new instance of {@code Gradients}
   */
  @SuppressWarnings({"unchecked", "rawtypes"})
  public static Gradients create(Scope scope, Operand<?> y, Iterable<Operand<?>> x, Options... options) {
    return create(scope, (Iterable) Arrays.asList(y), x, options);
  }

  /**
   * @param dx partial derivatives of some loss function {@code L} w.r.t. {@code y}
   * @return builder to add more options to this operation
   */
  public Options dx(Iterable<Operand<?>> dx) {
    return new Options().dx(dx);
  }

  @Override
  @SuppressWarnings({"rawtypes", "unchecked"})
  public Iterator<Operand<?>> iterator() {
    return (Iterator) dy.iterator();
  }
  
  /**
   * Partial derivatives of {@code y}s w.r.t. {@code x}s, with the size of {@code x}
   */
  public List<Output<?>> dy() {
    return dy;
  }
  
  /**
   * Returns a symbolic handle to one of the gradient operation output
   * <p>
   * Warning: Does not check that the type of the tensor matches T. It is recommended to call
   * this method with an explicit type parameter rather than letting it be inferred, e.g. {@code
   * gradients.<Integer>dy(0)}
   *
   * @param <T> The expected element type of the tensors produced by this output.
   * @param index The index of the output among the gradients added by this operation
   */
  @SuppressWarnings("unchecked")
  public <T> Output<T> dy(int index) {
    return (Output<T>) dy.get(index);
  }

  private List<Output<?>> dy;
  
  private Gradients(List<Output<?>> dy) {
    this.dy = dy;
  }
}
