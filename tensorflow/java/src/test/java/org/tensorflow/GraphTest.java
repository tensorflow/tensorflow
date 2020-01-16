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
      Output<Float> y1 = TestUtil.square(g, "y1", y0);
      Output<Float> y2 = TestUtil.addN(g, y0, x2);
      
      Output<?>[] grads0 = g.addGradients(y1, toArray(x1));
      assertNotNull(grads0);
      assertEquals(1, grads0.length);
      assertEquals(DataType.FLOAT, grads0[0].dataType());

      Output<?>[] grads1 = g.addGradients(y2, toArray(x1, x2));
      assertNotNull(grads1);
      assertEquals(2, grads1.length);
      assertEquals(DataType.FLOAT, grads1[0].dataType());
      assertEquals(DataType.FLOAT, grads1[1].dataType());
      
      try (Tensor<Float> c1 = Tensors.create(3.0f);
          Tensor<Float> c2 = Tensors.create(2.0f);
          TestUtil.AutoCloseableList<Tensor<?>> outputs = new TestUtil.AutoCloseableList<>(
              s.runner()
                  .feed(x1, c1)
                  .feed(x2, c2)
                  .fetch(grads0[0])
                  .fetch(grads1[0])
                  .fetch(grads1[1])
                  .run())) {
     
        assertEquals(3, outputs.size());
        assertEquals(108.0f, outputs.get(0).floatValue(), 0.0f);
        assertEquals(6.0f, outputs.get(1).floatValue(), 0.0f);
        assertEquals(1.0f, outputs.get(2).floatValue(), 0.0f);
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

      Output<?>[] grad = g.addGradients(null, toArray(y0, y1), toArray(x), null);
      assertNotNull(grad);
      assertEquals(1, grad.length);
      assertEquals(DataType.FLOAT, grad[0].dataType());

      try (Tensor<Float> c = Tensors.create(3.0f);
          Tensor<?> output = s.runner()
              .feed(x, c)
              .fetch(grad[0])
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
      Output<Float> y0 = TestUtil.square(g, "y0", x);
      Output<Float> y1 = TestUtil.square(g, "y1", y0);
      
      Output<?>[] grad0 = g.addGradients(y1, toArray(y0));
      assertNotNull(grad0);
      assertEquals(1, grad0.length);
      assertEquals(DataType.FLOAT, grad0[0].dataType());

      Output<?>[] grad1 = g.addGradients(null, toArray(y0), toArray(x), toArray(grad0[0]));
      assertNotNull(grad1);
      assertEquals(1, grad1.length);
      assertEquals(DataType.FLOAT, grad1[0].dataType());

      try (Tensor<Float> c = Tensors.create(3.0f);
          Tensor<?> output = s.runner()
              .feed(x, c)
              .fetch(grad1[0])
              .run()
              .get(0)) {
     
        assertEquals(108.0f, output.floatValue(), 0.0f);
      }
    }
  }

  @Test
  public void validateGradientsNames() {
    try (Graph g = new Graph()) {

      Output<Float> x = TestUtil.placeholder(g, "x", Float.class);
      Output<Float> y0 = TestUtil.square(g, "y0", x);

      Output<?>[] grad0 = g.addGradients(null, toArray(y0), toArray(x), null);
      assertTrue(grad0[0].op().name().startsWith("gradients/"));

      Output<?>[] grad1 = g.addGradients(null, toArray(y0), toArray(x), null);
      assertTrue(grad1[0].op().name().startsWith("gradients_1/"));

      Output<?>[] grad2 = g.addGradients("more_gradients", toArray(y0), toArray(x), null);
      assertTrue(grad2[0].op().name().startsWith("more_gradients/"));

      Output<?>[] grad3 = g.addGradients("even_more_gradients", toArray(y0), toArray(x), null);
      assertTrue(grad3[0].op().name().startsWith("even_more_gradients/"));

      try {
        g.addGradients("even_more_gradients", toArray(y0), toArray(x), null);
      } catch (IllegalArgumentException e) {
        // expected exception
      }
    }
  }

  @Test
  public void buildWhileLoopSingleInput() {
    try (Graph g = new Graph();
        Session s = new Session(g)) {

      Output<?> input = TestUtil.placeholder(g, "input1", Integer.class);

      // could write this using lambda after Java 8
      Graph.WhileSubgraphBuilder condGraphBuilder =
          new Graph.WhileSubgraphBuilder() {
            @Override
            public void buildSubgraph(
                Graph condGraph, Output<?>[] condInputs, Output<?>[] condOutputs) {
              Output<Integer> sixteen = TestUtil.constant(condGraph, "sixteen", 16);
              // condInputs[0] < 16
              Output<?> condOutput =
                  condGraph
                      .opBuilder("Less", "cond")
                      .addInput(condInputs[0])
                      .addInput(sixteen)
                      .build()
                      .output(0);

              condOutputs[0] = condOutput;
            }
          };

      // could write this using lambda after Java 8
      Graph.WhileSubgraphBuilder bodyGraphBuilder =
          new Graph.WhileSubgraphBuilder() {
            @Override
            public void buildSubgraph(
                Graph bodyGraph, Output<?>[] bodyInputs, Output<?>[] bodyOutputs) {
              bodyOutputs[0] = TestUtil.square(bodyGraph, "square", bodyInputs[0]);
            }
          };

      Output<?>[] loopOutputs =
          g.whileLoop(toArray(input), condGraphBuilder, bodyGraphBuilder, "test_loop");

      try (Tensor<Integer> c = Tensors.create(2);
          Tensor<?> output = s.runner().feed(input, c).fetch(loopOutputs[0]).run().get(0)) {

        assertEquals(16, output.intValue()); // ((2^2)^2)
      }
    }
  }

  @Test
  public void buildWhileLoopMultipleInputs() {
    try (Graph g = new Graph();
        Session s = new Session(g)) {

      Output<?> input1 = TestUtil.placeholder(g, "input1", Integer.class);
      Output<?> input2 = TestUtil.placeholder(g, "input2", Integer.class);
      Output<?>[] inputs = toArray(input1, input2);

      // could write this using lambda after Java 8
      Graph.WhileSubgraphBuilder condGraphBuilder =
          new Graph.WhileSubgraphBuilder() {
            @Override
            public void buildSubgraph(
                Graph condGraph, Output<?>[] condInputs, Output<?>[] condOutputs) {
              Output<Integer> sixteen = TestUtil.constant(condGraph, "sixteen", 16);
              Output<?> condOutput =
                  condGraph
                      .opBuilder("Less", "cond")
                      .addInput(condInputs[0])
                      .addInput(sixteen)
                      .build()
                      .output(0); // condInputs[0] < 16

              condOutputs[0] = condOutput;
            }
          };

      // could write this using lambda after Java 8
      Graph.WhileSubgraphBuilder bodyGraphBuilder =
          new Graph.WhileSubgraphBuilder() {
            @Override
            public void buildSubgraph(
                Graph bodyGraph, Output<?>[] bodyInputs, Output<?>[] bodyOutputs) {
              bodyOutputs[0] = TestUtil.square(bodyGraph, "square1", bodyInputs[0]);
              bodyOutputs[1] = TestUtil.square(bodyGraph, "square2", bodyInputs[1]);
            }
          };

      Output<?>[] loopOutputs =
          g.whileLoop(inputs, condGraphBuilder, bodyGraphBuilder, "test_loop");

      try (Tensor<Integer> c1 = Tensors.create(2);
          Tensor<Integer> c2 = Tensors.create(5);
          TestUtil.AutoCloseableList<Tensor<?>> outputs =
              new TestUtil.AutoCloseableList<>(
                  s.runner()
                      .feed(input1, c1)
                      .feed(input2, c2)
                      .fetch(loopOutputs[0])
                      .fetch(loopOutputs[1])
                      .run())) {

        assertEquals(2, outputs.size());
        assertEquals(16, outputs.get(0).intValue()); // ((2^2)^2)
        assertEquals(625, outputs.get(1).intValue()); // ((5^2)^2)
      }
    }
  }

  private static Output<?>[] toArray(Output<?>... outputs) {
    return outputs;
  }
}
