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
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.Operation}. */
@RunWith(JUnit4.class)
public class OperationTest {

  @Test
  public void outputListLengthFailsOnInvalidName() {
    try (Graph g = new Graph()) {
      Operation op =
          g.opBuilder("Add", "Add")
              .addInput(TestUtil.constant(g, "x", 1))
              .addInput(TestUtil.constant(g, "y", 2))
              .build();
      assertEquals(1, op.outputListLength("z"));

      try {
        op.outputListLength("unknown");
        fail("Did not catch bad name");
      } catch (IllegalArgumentException iae) {
        // expected
      }
    }
  }

  @Test
  public void operationEquality() {
    Operation op1;
    try (Graph g = new Graph()) {
      op1 = TestUtil.constant(g, "op1", 1).op();
      Operation op2 = TestUtil.constant(g, "op2", 2).op();
      Operation op3 = new Operation(g, op1.getUnsafeNativeHandle());
      Operation op4 = g.operation("op1");
      assertEquals(op1, op1);
      assertNotEquals(op1, op2);
      assertEquals(op1, op3);
      assertEquals(op1.hashCode(), op3.hashCode());
      assertEquals(op1, op4);
      assertEquals(op1.hashCode(), op4.hashCode());
      assertEquals(op3, op4);
      assertNotEquals(op2, op3);
      assertNotEquals(op2, op4);
    }
    try (Graph g = new Graph()) {
      Operation newOp1 = TestUtil.constant(g, "op1", 1).op();
      assertNotEquals(op1, newOp1);
    }
  }

  @Test
  public void operationCollection() {
    try (Graph g = new Graph()) {
      Operation op1 = TestUtil.constant(g, "op1", 1).op();
      Operation op2 = TestUtil.constant(g, "op2", 2).op();
      Operation op3 = new Operation(g, op1.getUnsafeNativeHandle());
      Operation op4 = g.operation("op1");
      Set<Operation> ops = new HashSet<>();
      ops.addAll(Arrays.asList(op1, op2, op3, op4));
      assertEquals(2, ops.size());
      assertTrue(ops.contains(op1));
      assertTrue(ops.contains(op2));
      assertTrue(ops.contains(op3));
      assertTrue(ops.contains(op4));
    }
  }

  @Test
  public void operationToString() {
    try (Graph g = new Graph()) {
      Operation op = TestUtil.constant(g, "c", new int[] {1}).op();
      assertNotNull(op.toString());
    }
  }

  @Test
  public void outputEquality() {
    try (Graph g = new Graph()) {
      Output<Integer> output = TestUtil.constant(g, "c", 1);
      Output<Integer> output1 = output.op().<Integer>output(0);
      Output<Integer> output2 = g.operation("c").<Integer>output(0);
      assertEquals(output, output1);
      assertEquals(output.hashCode(), output1.hashCode());
      assertEquals(output, output2);
      assertEquals(output.hashCode(), output2.hashCode());
    }
  }

  @Test
  public void outputCollection() {
    try (Graph g = new Graph()) {
      Output<Integer> output = TestUtil.constant(g, "c", 1);
      Output<Integer> output1 = output.op().<Integer>output(0);
      Output<Integer> output2 = g.operation("c").<Integer>output(0);
      Set<Output<Integer>> ops = new HashSet<>();
      ops.addAll(Arrays.asList(output, output1, output2));
      assertEquals(1, ops.size());
      assertTrue(ops.contains(output));
      assertTrue(ops.contains(output1));
      assertTrue(ops.contains(output2));
    }
  }

  @Test
  public void outputToString() {
    try (Graph g = new Graph()) {
      Output<Integer> output = TestUtil.constant(g, "c", new int[] {1});
      assertNotNull(output.toString());
    }
  }

  @Test
  public void outputListLength() {
    assertEquals(1, split(new int[] {0, 1}, 1));
    assertEquals(2, split(new int[] {0, 1}, 2));
    assertEquals(3, split(new int[] {0, 1, 2}, 3));
  }

  @Test
  public void inputListLength() {
    assertEquals(1, splitWithInputList(new int[] {0, 1}, 1, "split_dim"));
    try {
      splitWithInputList(new int[] {0, 1}, 2, "inputs");
    } catch (IllegalArgumentException iae) {
      // expected
    }
  }

  @Test
  public void outputList() {
    try (Graph g = new Graph()) {
      Operation split = TestUtil.split(g, "split", new int[] {0, 1, 2}, 3);
      Output<?>[] outputs = split.outputList(1, 2);
      assertNotNull(outputs);
      assertEquals(2, outputs.length);
      for (int i = 0; i < outputs.length; ++i) {
        assertEquals(i + 1, outputs[i].index());
      }
    }
  }

  private static int split(int[] values, int num_split) {
    try (Graph g = new Graph()) {
      return g.opBuilder("Split", "Split")
          .addInput(TestUtil.constant(g, "split_dim", 0))
          .addInput(TestUtil.constant(g, "values", values))
          .setAttr("num_split", num_split)
          .build()
          .outputListLength("output");
    }
  }

  private static int splitWithInputList(int[] values, int num_split, String name) {
    try (Graph g = new Graph()) {
      return g.opBuilder("Split", "Split")
          .addInput(TestUtil.constant(g, "split_dim", 0))
          .addInput(TestUtil.constant(g, "values", values))
          .setAttr("num_split", num_split)
          .build()
          .inputListLength(name);
    }
  }
}
