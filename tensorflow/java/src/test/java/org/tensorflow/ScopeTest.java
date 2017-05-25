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

package org.tensorflow;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.Scope}. */
@RunWith(JUnit4.class)
public class ScopeTest {

  @Test
  public void basic() {
    try (Graph g = new Graph()) {
      Scope s = Scope.create(g);
      Const c1 = Const.create(s, 42);
      assertEquals("Const", c1.input().op().name());
      Const c2 = Const.create(s, 7);
      assertEquals("Const_1", c2.input().op().name());
      Const c3 = Const.create(s.withOpName("four"), 4);
      assertEquals("four", c3.input().op().name());
    }
  }

  @Test
  public void hierarchy() {
    try (Graph g = new Graph()) {
      Scope root = Scope.create(g);
      Scope child = root.withSubScope("child");
      assertEquals("child/Const", Const.create(child, 42).input().op().name());
      assertEquals("child/four", Const.create(child.withOpName("four"), 4).input().op().name());
    }
  }

  @Test
  public void composite() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope s = Scope.create(g);
      Const data = Const.create(s.withOpName("data"), new int[] {600, 470, 170, 430, 300});

      // Create a composite op with a customized name
      Variance var1 = Variance.create(s.withOpName("example"), data);
      assertEquals("example/variance", var1.input().op().name());

      // Confirm internally added ops have the right names.
      assertNotNull(g.operation("example/squared_deviation"));
      assertNotNull(g.operation("example/Mean"));
      assertNotNull(g.operation("example/zero"));

      // Same composite op with a default name
      Variance var2 = Variance.create(s, data);
      assertEquals("variance/variance", var2.input().op().name());

      // Confirm internally added ops have the right names.
      assertNotNull(g.operation("variance/squared_deviation"));
      assertNotNull(g.operation("variance/Mean"));
      assertNotNull(g.operation("variance/zero"));

      // Verify correct results as well.
      Tensor result = sess.runner().fetch(var1.input()).run().get(0);
      assertEquals(21704, result.intValue());
      result = sess.runner().fetch(var2.input()).run().get(0);
      assertEquals(21704, result.intValue());
    }
  }

  // Convenience interface
  // TODO: replace when standardized
  interface Input {
    Output input();
  }

  // "handwritten" sample operator classes
  private static final class Const implements Input {
    private final Output output;

    private static Const create(Scope s, Object v) {
      try (Tensor value = Tensor.create(v)) {
        return new Const(
            s.graph()
                .opBuilder("Const", s.makeOpName("Const"))
                .setAttr("dtype", value.dataType())
                .setAttr("value", value)
                .build()
                .output(0));
      }
    }

    private Const(Output o) {
      output = o;
    }

    @Override
    public Output input() {
      return output;
    }
  }

  private static final class Mean implements Input {
    private final Output output;

    private static Mean create(Scope s, Input input, Input reductionIndices) {
      return new Mean(
          s.graph()
              .opBuilder("Mean", s.makeOpName("Mean"))
              .addInput(input.input())
              .addInput(reductionIndices.input())
              .build()
              .output(0));
    }

    private Mean(Output o) {
      output = o;
    }

    @Override
    public Output input() {
      return output;
    }
  }

  private static final class SquaredDifference implements Input {
    private final Output output;

    private static SquaredDifference create(Scope s, Input x, Input y) {
      return new SquaredDifference(
          s.graph()
              .opBuilder("SquaredDifference", s.makeOpName("SquaredDifference"))
              .addInput(x.input())
              .addInput(y.input())
              .build()
              .output(0));
    }

    private SquaredDifference(Output o) {
      output = o;
    }

    @Override
    public Output input() {
      return output;
    }
  }

  private static final class Variance implements Input {
    private final Output output;

    private static Variance create(Scope base, Input x) {
      Scope s = base.withSubScope("variance");
      Const zero = Const.create(s.withOpName("zero"), new int[] {0});
      SquaredDifference sqdiff =
          SquaredDifference.create(s.withOpName("squared_deviation"), x, Mean.create(s, x, zero));

      return new Variance(Mean.create(s.withOpName("variance"), sqdiff, zero).input());
    }

    private Variance(Output o) {
      output = o;
    }

    @Override
    public Output input() {
      return output;
    }
  }
}
