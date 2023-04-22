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
import static org.junit.Assert.assertNotEquals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link Shape}. */
@RunWith(JUnit4.class)
public class ShapeTest {

  @Test
  public void unknown() {
    assertEquals(-1, Shape.unknown().numDimensions());
    assertEquals("<unknown>", Shape.unknown().toString());
  }

  @Test
  public void scalar() {
    assertEquals(0, Shape.scalar().numDimensions());
    assertEquals("[]", Shape.scalar().toString());
  }

  @Test
  public void make() {
    Shape s = Shape.make(2);
    assertEquals(1, s.numDimensions());
    assertEquals(2, s.size(0));
    assertEquals("[2]", s.toString());

    s = Shape.make(2, 3);
    assertEquals(2, s.numDimensions());
    assertEquals(2, s.size(0));
    assertEquals(3, s.size(1));
    assertEquals("[2, 3]", s.toString());

    s = Shape.make(-1, 2, 3);
    assertEquals(3, s.numDimensions());
    assertEquals(-1, s.size(0));
    assertEquals(2, s.size(1));
    assertEquals(3, s.size(2));
    assertEquals("[?, 2, 3]", s.toString());
  }

  @Test
  public void nodesInAGraph() {
    try (Graph g = new Graph()) {
      Output<Float> n = TestUtil.placeholder(g, "feed", Float.class);
      assertEquals(-1, n.shape().numDimensions());

      n = TestUtil.constant(g, "scalar", 3);
      assertEquals(0, n.shape().numDimensions());

      n = TestUtil.constant(g, "vector", new float[2]);
      assertEquals(1, n.shape().numDimensions());
      assertEquals(2, n.shape().size(0));

      n = TestUtil.constant(g, "matrix", new float[4][5]);
      assertEquals(2, n.shape().numDimensions());
      assertEquals(4, n.shape().size(0));
      assertEquals(5, n.shape().size(1));
    }
  }

  @Test
  public void equalsWorksCorrectly() {
    assertEquals(Shape.scalar(), Shape.scalar());
    assertEquals(Shape.make(1, 2, 3), Shape.make(1, 2, 3));

    assertNotEquals(Shape.make(1, 2), null);
    assertNotEquals(Shape.make(1, 2), new Object());
    assertNotEquals(Shape.make(1, 2, 3), Shape.make(1, 2, 4));

    assertNotEquals(Shape.unknown(), Shape.unknown());
    assertNotEquals(Shape.make(-1), Shape.make(-1));
    assertNotEquals(Shape.make(1, -1, 3), Shape.make(1, -1, 3));
  }

  @Test
  public void hashCodeIsAsExpected() {
    assertEquals(Shape.make(1, 2, 3, 4).hashCode(), Shape.make(1, 2, 3, 4).hashCode());
    assertEquals(Shape.scalar().hashCode(), Shape.scalar().hashCode());
    assertEquals(Shape.unknown().hashCode(), Shape.unknown().hashCode());

    assertNotEquals(Shape.make(1, 2).hashCode(), Shape.make(1, 3).hashCode());
  }
}
