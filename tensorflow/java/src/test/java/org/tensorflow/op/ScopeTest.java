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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/** Unit tests for {@link org.tensorflow.Scope}. */
@RunWith(JUnit4.class)
public class ScopeTest {

  @Test
  public void basicNames() {
    try (Graph g = new Graph()) {
      Scope root = new Scope(g);
      assertEquals("add", root.makeOpName("add"));
      assertEquals("add_1", root.makeOpName("add"));
      assertEquals("add_2", root.makeOpName("add"));
      assertEquals("mul", root.makeOpName("mul"));
    }
  }

  @Test
  public void hierarchicalNames() {
    try (Graph g = new Graph()) {
      Scope root = new Scope(g);
      Scope child = root.withSubScope("child");
      assertEquals("child/add", child.makeOpName("add"));
      assertEquals("child/add_1", child.makeOpName("add"));
      assertEquals("child/mul", child.makeOpName("mul"));

      Scope child_1 = root.withSubScope("child");
      assertEquals("child_1/add", child_1.makeOpName("add"));
      assertEquals("child_1/add_1", child_1.makeOpName("add"));
      assertEquals("child_1/mul", child_1.makeOpName("mul"));

      Scope c_c = root.withSubScope("c").withSubScope("c");
      assertEquals("c/c/add", c_c.makeOpName("add"));

      Scope c_1 = root.withSubScope("c");
      Scope c_1_c = c_1.withSubScope("c");
      assertEquals("c_1/c/add", c_1_c.makeOpName("add"));

      Scope c_1_c_1 = c_1.withSubScope("c");
      assertEquals("c_1/c_1/add", c_1_c_1.makeOpName("add"));
    }
  }

  @Test
  public void scopeAndOpNames() {
    try (Graph g = new Graph()) {
      Scope root = new Scope(g);

      Scope child = root.withSubScope("child");

      assertEquals("child/add", child.makeOpName("add"));
      assertEquals("child_1", root.makeOpName("child"));
      assertEquals("child_2/p", root.withSubScope("child").makeOpName("p"));
    }
  }

  @Test
  public void validateNames() {
    try (Graph g = new Graph()) {
      Scope root = new Scope(g);

      final String[] invalid_names = {
        "_", "-", "-x", // Names are constrained to start with [A-Za-z0-9.]
        null, "", "a$", // Invalid characters
        "a/b", // slashes not allowed
      };

      for (String name : invalid_names) {
        try {
          root.withName(name);
          fail("failed to catch invalid op name.");
        } catch (IllegalArgumentException ex) {
          // expected
        }
        // Subscopes follow the same rules
        try {
          root.withSubScope(name);
          fail("failed to catch invalid scope name: " + name);
        } catch (IllegalArgumentException ex) {
          // expected
        }
      }

      // Unusual but valid names.
      final String[] valid_names = {".", "..", "._-.", "a--."};

      for (String name : valid_names) {
        root.withName(name);
        root.withSubScope(name);
      }
    }
  }

  @Test
  public void basic() {
    try (Graph g = new Graph()) {
      Scope s = new Scope(g);
      Const c1 = Const.create(s, 42);
      assertEquals("Const", c1.output().op().name());
      Const c2 = Const.create(s, 7);
      assertEquals("Const_1", c2.output().op().name());
      Const c3 = Const.create(s.withName("four"), 4);
      assertEquals("four", c3.output().op().name());
      Const c4 = Const.create(s.withName("four"), 4);
      assertEquals("four_1", c4.output().op().name());
    }
  }

  @Test
  public void hierarchy() {
    try (Graph g = new Graph()) {
      Scope root = new Scope(g);
      Scope child = root.withSubScope("child");
      assertEquals("child/Const", Const.create(child, 42).output().op().name());
      assertEquals("child/four", Const.create(child.withName("four"), 4).output().op().name());
    }
  }

  @Test
  public void composite() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope s = new Scope(g);
      Output data = Const.create(s.withName("data"), new int[] {600, 470, 170, 430, 300}).output();

      // Create a composite op with a customized name
      Variance var1 = Variance.create(s.withName("example"), data);
      assertEquals("example/variance", var1.output().op().name());

      // Confirm internally added ops have the right names.
      assertNotNull(g.operation("example/squared_deviation"));
      assertNotNull(g.operation("example/Mean"));
      assertNotNull(g.operation("example/zero"));

      // Same composite op with a default name
      Variance var2 = Variance.create(s, data);
      assertEquals("variance/variance", var2.output().op().name());

      // Confirm internally added ops have the right names.
      assertNotNull(g.operation("variance/squared_deviation"));
      assertNotNull(g.operation("variance/Mean"));
      assertNotNull(g.operation("variance/zero"));

      // Verify correct results as well.
      Tensor result = sess.runner().fetch(var1.output()).run().get(0);
      assertEquals(21704, result.intValue());
      result = sess.runner().fetch(var2.output()).run().get(0);
      assertEquals(21704, result.intValue());
    }
  }

  // "handwritten" sample operator classes
  private static final class Const {
    private final Output output;

    static Const create(Scope s, Object v) {
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

    Const(Output o) {
      output = o;
    }

    Output output() {
      return output;
    }
  }

  private static final class Mean {
    private final Output output;

    static Mean create(Scope s, Output input, Output reductionIndices) {
      return new Mean(
          s.graph()
              .opBuilder("Mean", s.makeOpName("Mean"))
              .addInput(input)
              .addInput(reductionIndices)
              .build()
              .output(0));
    }

    Mean(Output o) {
      output = o;
    }

    Output output() {
      return output;
    }
  }

  private static final class SquaredDifference {
    private final Output output;

    static SquaredDifference create(Scope s, Output x, Output y) {
      return new SquaredDifference(
          s.graph()
              .opBuilder("SquaredDifference", s.makeOpName("SquaredDifference"))
              .addInput(x)
              .addInput(y)
              .build()
              .output(0));
    }

    SquaredDifference(Output o) {
      output = o;
    }

    Output output() {
      return output;
    }
  }

  private static final class Variance {
    private final Output output;

    static Variance create(Scope base, Output x) {
      Scope s = base.withSubScope("variance");
      Output zero = Const.create(s.withName("zero"), new int[] {0}).output();
      Output sqdiff =
          SquaredDifference.create(
                  s.withName("squared_deviation"), x, Mean.create(s, x, zero).output())
              .output();

      return new Variance(Mean.create(s.withName("variance"), sqdiff, zero).output());
    }

    Variance(Output o) {
      output = o;
    }

    Output output() {
      return output;
    }
  }
}
