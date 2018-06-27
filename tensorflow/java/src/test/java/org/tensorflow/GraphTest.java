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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.util.HashSet;
import java.util.Iterator;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.Graph}. */
@RunWith(JUnit4.class)
public class GraphTest {

  @Test
  public void graphDefRoundTrip() {
    byte[] graphDef;
    // Create a graph for A * X + B
    try (Graph g = new Graph()) {
      TestUtil.transpose_A_times_X(g, new int[2][2]);
      graphDef = g.toGraphDef();
    }
    // Import the GraphDef and find all the nodes.
    try (Graph g = new Graph()) {
      g.importGraphDef(graphDef);
      validateImportedGraph(g, "");
    }
    try (Graph g = new Graph()) {
      g.importGraphDef(graphDef, "BugsBunny");
      validateImportedGraph(g, "BugsBunny/");
    }
  }

  // Helper function whose implementation is based on knowledge of how
  // TestUtil.transpose_A_times_X is implemented.
  private static void validateImportedGraph(Graph g, String prefix) {
    Operation op = g.operation(prefix + "A");
    assertNotNull(op);
    assertEquals(prefix + "A", op.name());
    assertEquals("Const", op.type());
    assertEquals(1, op.numOutputs());
    assertEquals(op, op.output(0).op());

    op = g.operation(prefix + "X");
    assertNotNull(op);
    assertEquals(prefix + "X", op.name());
    assertEquals("Placeholder", op.type());
    assertEquals(1, op.numOutputs());
    assertEquals(op, op.output(0).op());

    op = g.operation(prefix + "Y");
    assertNotNull(op);
    assertEquals(prefix + "Y", op.name());
    assertEquals("MatMul", op.type());
    assertEquals(1, op.numOutputs());
    assertEquals(op, op.output(0).op());
  }

  @Test
  public void iterateOverOperations() {
    try (Graph g = new Graph()) {
      Iterator<Operation> iterator = g.operations();
      HashSet<Operation> operations;

      assertFalse(iterator.hasNext());

      operations = new HashSet<>();
      operations.add(TestUtil.constant(g, "Const-A", Float.valueOf(1.0f)).op());
      operations.add(TestUtil.constant(g, "Const-B", Integer.valueOf(23)).op());
      operations.add(TestUtil.constant(g, "Const-C", Double.valueOf(1.618)).op());

      iterator = g.operations();

      assertTrue(iterator.hasNext());
      assertTrue(operations.remove(iterator.next()));

      assertTrue(iterator.hasNext());
      assertTrue(operations.remove(iterator.next()));

      assertTrue(iterator.hasNext());
      assertTrue(operations.remove(iterator.next()));

      assertFalse(iterator.hasNext());
    }
  }

  @Test
  public void failImportOnInvalidGraphDefs() {
    try (Graph g = new Graph()) {
      try {
        g.importGraphDef(null);
      } catch (IllegalArgumentException e) {
        // expected exception.
      }

      try {
        g.importGraphDef(new byte[] {1});
      } catch (IllegalArgumentException e) {
        // expected exception.
      }
    }
  }

  @Test
  public void failOnUseAfterClose() {
    Graph g = new Graph();
    g.close();
    try {
      g.toGraphDef();
    } catch (IllegalStateException e) {
      // expected exception.
    }
  }

  @Test
  public void addGradientsToGraph() {
    try (Graph g = new Graph();
        Session s = new Session(g)) {

      Output<Float> x1 = TestUtil.placeholder(g, "x1", Float.class);
      Output<Float> x2 = TestUtil.placeholder(g, "x2", Float.class);
      Output<Float> y0 = TestUtil.square(g, "y0", x1);
      Output<Float> y1 = TestUtil.addN(g, y0, x2);
      
      Output<?>[] grads = g.addGradients(y1, toArray(x1, x2));
      assertNotNull(grads);
      assertEquals(2, grads.length);
      assertEquals(DataType.FLOAT, grads[0].dataType());
      assertEquals(DataType.FLOAT, grads[1].dataType());
      
      try (Tensor<Float> c1 = Tensors.create(3.0f);
          Tensor<Float> c2 = Tensors.create(2.0f);
          TestUtil.AutoCloseableList<Tensor<?>> outputs = new TestUtil.AutoCloseableList<>(
              s.runner()
                  .feed(x1, c1)
                  .feed(x2, c2)
                  .fetch(grads[0])
                  .fetch(grads[1])
                  .run())) {
     
        assertEquals(6.0f, outputs.get(0).floatValue(), 0.0f);
        assertEquals(1.0f, outputs.get(1).floatValue(), 0.0f);
      }
    }
  }

  @Test
  public void addGradientSumsToGraph() {
    try (Graph g = new Graph();
        Session s = new Session(g)) {

      Output<Float> x = TestUtil.placeholder(g, "x", Float.class);
      Output<Float> y0 = TestUtil.square(g, "y0", x);
      Output<Float> y1 = TestUtil.square(g, "y1", y0);
      
      Output<?>[] grads = g.addGradients(toArray(y0, y1), toArray(x), null);

      try (Tensor<Float> c = Tensors.create(3.0f);
          Tensor<?> output = s.runner()
              .feed(x, c)
              .fetch(grads[0])
              .run()
              .get(0)) {
     
        assertEquals(114.0f, output.floatValue(), 0.0f);
      }
    }
  }

  @Test
  public void addGradientsWithInitialValuesToGraph() {
    try (Graph g = new Graph();
        Session s = new Session(g)) {

      Output<Float> x = TestUtil.placeholder(g, "x", Float.class);
      Output<Float> y = TestUtil.square(g, "y", x);
      Output<Float> dx = TestUtil.constant(g, "dx", 18.0f);
      
      Output<?>[] grads = g.addGradients(toArray(y), toArray(x), toArray(dx));

      try (Tensor<Float> c = Tensors.create(3.0f);
          Tensor<?> output = s.runner()
              .feed(x, c)
              .fetch(grads[0])
              .run()
              .get(0)) {
     
        assertEquals(108.0f, output.floatValue(), 0.0f);
      }
    }
  }
  
  private static Output<?>[] toArray(Output<?>... outputs) {
    return outputs;
  }
}
