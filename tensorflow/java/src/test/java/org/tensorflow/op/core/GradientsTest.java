package org.tensorflow.op.core;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import org.tensorflow.TestUtil;
import org.tensorflow.op.Scope;

@RunWith(JUnit4.class)
public class GradientsTest {

  @Test
  public void createGradients() {
    try (Graph g = new Graph(); 
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);

      Output<Float> x = TestUtil.placeholder(g, "x1", Float.class);
      Output<Float> y0 = TestUtil.square(g, "y0", x);
      Output<Float> y1 = TestUtil.square(g, "y1", y0);
      
      Gradients grads = Gradients.create(scope, y1, Arrays.asList(x, y0));
      
      assertNotNull(grads);
      assertNotNull(grads.dy());
      assertEquals(2, grads.dy().size());

      try (Tensor<Float> c = Tensors.create(3.0f);
          TestUtil.AutoCloseableList<Tensor<?>> outputs = new TestUtil.AutoCloseableList<>(
              sess.runner()
                  .feed(x, c)
                  .fetch(grads.dy(0))
                  .fetch(grads.dy(1))
                  .run())) {
     
        assertEquals(108.0f, outputs.get(0).floatValue(), 0.0f);
        assertEquals(18.0f, outputs.get(1).floatValue(), 0.0f);
      }
    }
  }

  @Test
  public void createGradientsWithSum() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);

      Output<Float> x = TestUtil.placeholder(g, "x1", Float.class);
      Output<Float> y0 = TestUtil.square(g, "y0", x);
      Output<Float> y1 = TestUtil.square(g, "y1", y0);
      
      Gradients grads = Gradients.create(scope, Arrays.asList(y0, y1), Arrays.asList(x));
      
      assertNotNull(grads);
      assertNotNull(grads.dy());
      assertEquals(1, grads.dy().size());

      try (Tensor<Float> c = Tensors.create(3.0f);
          TestUtil.AutoCloseableList<Tensor<?>> outputs = new TestUtil.AutoCloseableList<>(
              sess.runner()
                  .feed(x, c)
                  .fetch(grads.dy(0))
                  .run())) {
     
        assertEquals(114.0f, outputs.get(0).floatValue(), 0.0f);
      }
    }
  }

  @Test
  public void createGradientsWithInitialValues() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);

      Output<Float> x = TestUtil.placeholder(g, "x1", Float.class);
      Output<Float> y0 = TestUtil.square(g, "y0", x);
      Output<Float> y1 = TestUtil.square(g, "y1", y0);
      
      Gradients grads0 = Gradients.create(scope, y1, Arrays.asList(y0));
      Gradients grads1 = Gradients.create(scope, y0, Arrays.asList(x), Gradients.dx(grads0.dy()));
      
      assertNotNull(grads1);
      assertNotNull(grads1.dy());
      assertEquals(1, grads1.dy().size());

      try (Tensor<Float> c = Tensors.create(3.0f);
          TestUtil.AutoCloseableList<Tensor<?>> outputs = new TestUtil.AutoCloseableList<>(
              sess.runner()
                  .feed(x, c)
                  .fetch(grads1.dy(0))
                  .run())) {
     
        assertEquals(108.0f, outputs.get(0).floatValue(), 0.0f);
      }
    }
  }

  @Test
  public void createGradientsWithScopeName() {
    try (Graph g = new Graph()) {
      Scope scope = new Scope(g);

      Output<Float> x = TestUtil.placeholder(g, "x1", Float.class);
      Output<Float> y = TestUtil.square(g, "y", x);
      
      Scope gradScope = scope.withSubScope("grads").withSubScope("test");
      Gradients grads = Gradients.create(gradScope, y, Arrays.asList(x));
      
      assertTrue(grads.dy(0).op().name().startsWith("grads/test/"));
    }
  }
}
