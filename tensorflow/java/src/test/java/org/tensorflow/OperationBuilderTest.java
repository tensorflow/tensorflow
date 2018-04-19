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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.OperationBuilder}. */
@RunWith(JUnit4.class)
public class OperationBuilderTest {
  // TODO(ashankar): Restore this test once the C API gracefully handles mixing graphs and
  // operations instead of segfaulting.
  @Test
  @Ignore
  public void failWhenMixingOperationsOnDifferentGraphs() {
    try (Graph g1 = new Graph();
        Graph g2 = new Graph()) {
      Output<Integer> c1 = TestUtil.constant(g1, "C1", 3);
      Output<Integer> c2 = TestUtil.constant(g2, "C2", 3);
      TestUtil.addN(g1, c1, c1);
      try {
        TestUtil.addN(g2, c1, c2);
      } catch (Exception e) {
        fail(e.toString());
      }
    }
  }

  @Test
  public void failOnUseAfterBuild() {
    try (Graph g = new Graph();
        Tensor<Integer> t = Tensors.create(1)) {
      OperationBuilder b =
          g.opBuilder("Const", "Const").setAttr("dtype", t.dataType()).setAttr("value", t);
      b.build();
      try {
        b.setAttr("dtype", t.dataType());
      } catch (IllegalStateException e) {
        // expected exception.
      }
    }
  }

  @Test
  public void failOnUseAfterGraphClose() {
    OperationBuilder b = null;
    try (Graph g = new Graph();
        Tensor<Integer> t = Tensors.create(1)) {
      b = g.opBuilder("Const", "Const").setAttr("dtype", t.dataType()).setAttr("value", t);
    }
    try {
      b.build();
    } catch (IllegalStateException e) {
      // expected exception.
    }
  }

  @Test
  public void setAttr() {
    // The effect of setting an attribute may not easily be visible from the other parts of this
    // package's API. Thus, for now, the test simply executes the various setAttr variants to see
    // that there are no exceptions. If an attribute is "visible", test for that in a separate test
    // (like setAttrShape).
    //
    // This is a bit of an awkward test since it has to find operations with attributes of specific
    // types that aren't inferred from the input arguments.
    try (Graph g = new Graph()) {
      // dtype, tensor attributes.
      try (Tensor<Integer> t = Tensors.create(1)) {
        g.opBuilder("Const", "DataTypeAndTensor")
            .setAttr("dtype", DataType.INT32)
            .setAttr("value", t)
            .build()
            .output(0);
        assertTrue(hasNode(g, "DataTypeAndTensor"));
      }
      // string, bool attributes.
      g.opBuilder("Abort", "StringAndBool")
          .setAttr("error_msg", "SomeErrorMessage")
          .setAttr("exit_without_error", false)
          .build();
      assertTrue(hasNode(g, "StringAndBool"));
      // int (TF "int" attributes are 64-bit signed, so a Java long).
      g.opBuilder("RandomUniform", "Int")
          .addInput(TestUtil.constant(g, "RandomUniformShape", new int[] {1}))
          .setAttr("seed", 10)
          .setAttr("dtype", DataType.FLOAT)
          .build();
      assertTrue(hasNode(g, "Int"));
      // list(int)
      g.opBuilder("MaxPool", "IntList")
          .addInput(TestUtil.constant(g, "MaxPoolInput", new float[2][2][2][2]))
          .setAttr("ksize", new long[] {1, 1, 1, 1})
          .setAttr("strides", new long[] {1, 1, 1, 1})
          .setAttr("padding", "SAME")
          .build();
      assertTrue(hasNode(g, "IntList"));
      // list(float)
      g.opBuilder("FractionalMaxPool", "FloatList")
          .addInput(TestUtil.constant(g, "FractionalMaxPoolInput", new float[2][2][2][2]))
          .setAttr("pooling_ratio", new float[] {1.0f, 1.44f, 1.73f, 1.0f})
          .build();
      assertTrue(hasNode(g, "FloatList"));
      // Missing tests: float, list(dtype), list(tensor), list(string), list(bool)
    }
  }

  @Test
  public void setAttrShape() {
    try (Graph g = new Graph()) {
      Output<?> n =
          g.opBuilder("Placeholder", "unknown")
              .setAttr("dtype", DataType.FLOAT)
              .setAttr("shape", Shape.unknown())
              .build()
              .output(0);
      assertEquals(-1, n.shape().numDimensions());
      assertEquals(DataType.FLOAT, n.dataType());

      n = g.opBuilder("Placeholder", "batch_of_vectors")
              .setAttr("dtype", DataType.FLOAT)
              .setAttr("shape", Shape.make(-1, 784))
              .build()
              .output(0);
      assertEquals(2, n.shape().numDimensions());
      assertEquals(-1, n.shape().size(0));
      assertEquals(784, n.shape().size(1));
      assertEquals(DataType.FLOAT, n.dataType());
    }
  }

  @Test
  public void setAttrShapeList() {
    // Those shapes match tensors ones, so no exception is thrown
    testSetAttrShapeList(new Shape[] {Shape.make(2, 2), Shape.make(2, 2, 2)});
    try {
      // Those shapes do not match tensors ones, exception is thrown
      testSetAttrShapeList(new Shape[] {Shape.make(2, 2), Shape.make(2, 2, 2, 2)});
      fail("Shapes are incompatible and an exception was expected");
    } catch (IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void addControlInput() {
    try (Graph g = new Graph();
        Session s = new Session(g);
        Tensor<Boolean> yes = Tensors.create(true);
        Tensor<Boolean> no = Tensors.create(false)) {
      Output<Boolean> placeholder = TestUtil.placeholder(g, "boolean", Boolean.class);
      Operation check =
          g.opBuilder("Assert", "assert")
              .addInput(placeholder)
              .addInputList(new Output<?>[] {placeholder})
              .build();
      Operation noop = g.opBuilder("NoOp", "noop").addControlInput(check).build();

      // No problems when the Assert check succeeds
      s.runner().feed(placeholder, yes).addTarget(noop).run();

      // Exception thrown by the execution of the Assert node
      try {
        s.runner().feed(placeholder, no).addTarget(noop).run();
        fail("Did not run control operation.");
      } catch (IllegalArgumentException e) {
        // expected
      }
    }
  }

  private static void testSetAttrShapeList(Shape[] shapes) {
    try (Graph g = new Graph();
        Session s = new Session(g)) {
      int[][] matrix = new int[][] {{0, 0}, {0, 0}};
      Output<?> queue =
          g.opBuilder("FIFOQueue", "queue")
              .setAttr("component_types", new DataType[] {DataType.INT32, DataType.INT32})
              .setAttr("shapes", shapes)
              .build()
              .output(0);
      assertTrue(hasNode(g, "queue"));
      Output<Integer> c1 = TestUtil.constant(g, "const1", matrix);
      Output<Integer> c2 = TestUtil.constant(g, "const2", new int[][][] {matrix, matrix});
      Operation enqueue =
          g.opBuilder("QueueEnqueue", "enqueue")
              .addInput(queue)
              .addInputList(new Output<?>[] {c1, c2})
              .build();
      assertTrue(hasNode(g, "enqueue"));

      s.runner().addTarget(enqueue).run();
    }
  }

  private static boolean hasNode(Graph g, String name) {
    return g.operation(name) != null;
  }
}
