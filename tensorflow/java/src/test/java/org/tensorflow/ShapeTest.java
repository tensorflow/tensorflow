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

import java.util.Arrays;

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
    Shape s1 = Shape.make(-1, 2, 3);
    Shape s2 = s1;
    Shape s3 = Shape.make(-1, 2, 3);
    Shape s4 = Shape.make(-1, 2, 4);
    Shape s5 = Shape.make(-1, 2, 3, 4);
    Shape s6 = Shape.make(-1, 2);
    Object o = new Object();

    assertEquals(s1, s2);
    assertEquals(s1, s3);
    assertNotEquals(s1, s4);
    assertNotEquals(s1, s5);
    assertNotEquals(s1, s6);
    assertNotEquals(s1, o);
    assertNotEquals(s1, null);
  }

  @Test
  public void hashCodeIsAsExpected() {
    long[] d1 = new long[] {1, 2, 3, 4};
    long[] d2 = new long[] {};

    Shape s1 = Shape.make(1, 2, 3, 4);
    Shape s2 = Shape.scalar();

    assertEquals(Arrays.hashCode(d1), s1.hashCode());
    assertEquals(Arrays.hashCode(d2), s2.hashCode());
  }
}

