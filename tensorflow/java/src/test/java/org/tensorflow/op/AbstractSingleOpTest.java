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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.junit.Test;
import org.tensorflow.Graph;
import org.tensorflow.Input;
import org.tensorflow.Output;
import org.tensorflow.TestUtil;

public class AbstractSingleOpTest {

  @Test
  public void createOp() {
    try (Graph g = new Graph()) {
      Scope s = new Scope(g);
      Output array = TestUtil.constant(g, "array", new int[2][2]);

      TestOp test = TestOp.create(s, array);
      assertNotNull(test.operation);

      Output result = test.output();
      assertNotNull(result);
      assertSame(test.operation, result.op());
      assertEquals(2, result.shape().size(0)); // 2 dims
    }
  }

  @Test
  public void createListOp() {
    try (Graph g = new Graph()) {
      Scope s = new Scope(g);
      Output matrix = TestUtil.constant(g, "matrix", new int[2][2]);
      Output array = TestUtil.constant(g, "array", new int[2]);

      TestListOp test = TestListOp.create(s, Arrays.asList(matrix, array));
      assertNotNull(test.operation);

      List<Output> result = test.output();
      assertNotNull(test.output());
      assertEquals(2, result.size());
      assertEquals(2, result.get(0).shape().size(0)); // 2 dims
      assertEquals(1, result.get(1).shape().size(0)); // 1 dim
    }
  }

  @Test
  public void overrideOpName() {
    try (Graph g = new Graph()) {
      Scope s = new Scope(g);
      Output array = TestUtil.constant(g, "array", new int[2]);

      TestOp test1 = TestOp.create(s, array);
      assertEquals("shape", test1.operation.name());

      TestOp test2 = TestOp.create(s.withName("Test"), array);
      assertEquals("Test", test2.operation.name());
    }
  }

  @Test
  public void equalsHashcode() {
    try (Graph g = new Graph()) {
      Scope s = new Scope(g);
      Output array = TestUtil.constant(g, "array", new int[2]);

      TestOp test1 = TestOp.create(s, array);
      TestOp test2 = TestOp.create(s, array);
      AbstractSingleOp test3 = new AbstractSingleOp(test1.operation) {};

      // equals() tests
      assertNotEquals(test1, test2);
      assertEquals(test1, test3);
      assertEquals(test3, test1);
      assertNotEquals(test2, test3);

      // hashcode() tests
      Set<AbstractSingleOp> ops = new HashSet<>();
      assertTrue(ops.add(test1));
      assertTrue(ops.add(test2));
      assertFalse(ops.add(test3));
    }
  }

  @Test
  public void getOpString() {
    try (Graph g = new Graph()) {
      Scope s = new Scope(g);
      Output array = TestUtil.constant(g, "array", new int[2]);

      TestOp test = TestOp.create(s.withSubScope("tests"), array);
      assertTrue(test.toString().contains("tests/shape"));
      assertTrue(test.toString().contains("Shape"));
    }
  }

  private static class TestOp extends AbstractSingleOp implements Input {

    @Override
    public Output asOutput() {
      return output();
    }

    Output output() {
      return output;
    }

    static TestOp create(Scope s, Input input) {
      OperationHelper shape = OperationHelper.create(s, "Shape");
      shape.builder().addInput(input.asOutput());
      return new TestOp(shape);
    }

    private Output output;

    private TestOp(OperationHelper shape) {
      super(shape.operation());
      output = shape.nextOutput();
    }
  }

  private static class TestListOp extends AbstractSingleOp implements Iterable<Output> {

    @Override
    public Iterator<Output> iterator() {
      return output.iterator();
    }

    List<Output> output() {
      return output;
    }

    static TestListOp create(Scope s, Iterable<? extends Input> input) {
      OperationHelper shapeN = OperationHelper.create(s, "ShapeN");
      shapeN.builder().addInputList(Inputs.asOutputs(input));
      return new TestListOp(shapeN);
    }

    private List<Output> output;

    private TestListOp(OperationHelper shapeN) {
      super(shapeN.operation());
      output = shapeN.nextOutputList("output");
    }
  }
}
