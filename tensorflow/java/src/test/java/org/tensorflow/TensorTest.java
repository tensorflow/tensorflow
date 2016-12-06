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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.Tensor}. */
@RunWith(JUnit4.class)
public class TensorTest {
  @Test
  public void scalars() {
    try (Tensor t = Tensor.create(2.718f)) {
      assertEquals(DataType.FLOAT, t.dataType());
      assertEquals(0, t.numDimensions());
      assertEquals(0, t.shape().length);
      assertEquals(2.718f, t.floatValue(), 0);
    }

    try (Tensor t = Tensor.create(3.1415)) {
      assertEquals(DataType.DOUBLE, t.dataType());
      assertEquals(0, t.numDimensions());
      assertEquals(0, t.shape().length);
      assertEquals(3.1415, t.doubleValue(), 0);
    }

    try (Tensor t = Tensor.create(-33)) {
      assertEquals(DataType.INT32, t.dataType());
      assertEquals(0, t.numDimensions());
      assertEquals(0, t.shape().length);
      assertEquals(-33, t.intValue());
    }

    try (Tensor t = Tensor.create(8589934592L)) {
      assertEquals(DataType.INT64, t.dataType());
      assertEquals(0, t.numDimensions());
      assertEquals(0, t.shape().length);
      assertEquals(8589934592L, t.longValue());
    }

    try (Tensor t = Tensor.create(true)) {
      assertEquals(DataType.BOOL, t.dataType());
      assertEquals(0, t.numDimensions());
      assertEquals(0, t.shape().length);
      assertTrue(t.booleanValue());
    }

    final byte[] bytes = {1,2,3,4};
    try (Tensor t = Tensor.create(bytes)) {
      assertEquals(DataType.STRING, t.dataType());
      assertEquals(0, t.numDimensions());
      assertEquals(0, t.shape().length);
      assertArrayEquals(bytes, t.bytesValue());
    }
  }

  @Test
  public void nDimensional() {
    double[] vector = {1.414, 2.718, 3.1415};
    try (Tensor t = Tensor.create(vector)) {
      assertEquals(DataType.DOUBLE, t.dataType());
      assertEquals(1, t.numDimensions());
      assertArrayEquals(new long[] {3}, t.shape());

      double[] got = new double[3];
      assertArrayEquals(vector, t.copyTo(got), 0);
    }

    int[][] matrix = {{1, 2, 3}, {4, 5, 6}};
    try (Tensor t = Tensor.create(matrix)) {
      assertEquals(DataType.INT32, t.dataType());
      assertEquals(2, t.numDimensions());
      assertArrayEquals(new long[] {2, 3}, t.shape());

      int[][] got = new int[2][3];
      assertArrayEquals(matrix, t.copyTo(got));
    }

    long[][][] threeD = {
      {{1}, {3}, {5}, {7}, {9}}, {{2}, {4}, {6}, {8}, {0}},
    };
    try (Tensor t = Tensor.create(threeD)) {
      assertEquals(DataType.INT64, t.dataType());
      assertEquals(3, t.numDimensions());
      assertArrayEquals(new long[] {2, 5, 1}, t.shape());

      long[][][] got = new long[2][5][1];
      assertArrayEquals(threeD, t.copyTo(got));
    }

    boolean[][][][] fourD = {
      {{{false, false, false, true}, {false, false, true, false}}},
      {{{false, false, true, true}, {false, true, false, false}}},
      {{{false, true, false, true}, {false, true, true, false}}},
    };
    try (Tensor t = Tensor.create(fourD)) {
      assertEquals(DataType.BOOL, t.dataType());
      assertEquals(4, t.numDimensions());
      assertArrayEquals(new long[] {3, 1, 2, 4}, t.shape());

      boolean[][][][] got = new boolean[3][1][2][4];
      assertArrayEquals(fourD, t.copyTo(got));
    }
  }

  @Test
  public void failCreateOnMismatchedDimensions() {
    int[][][] invalid = new int[3][1][];
    for (int x = 0; x < invalid.length; ++x) {
      for (int y = 0; y < invalid[x].length; ++y) {
        invalid[x][y] = new int[x + y + 1];
      }
    }
    try (Tensor t = Tensor.create(invalid)) {
      fail("Tensor.create() should fail because of differing sizes in the 3rd dimension");
    } catch (IllegalArgumentException e) {
      // The expected exception.
    }
  }

  @Test
  public void failCopyToOnIncompatibleDestination() {
    try (final Tensor matrix = Tensor.create(new int[][] {{1, 2}, {3, 4}})) {
      try {
        matrix.copyTo(new int[2]);
        fail("should have failed on dimension mismatch");
      } catch (IllegalArgumentException e) {
        // The expected exception.
      }

      try {
        matrix.copyTo(new float[2][2]);
        fail("should have failed on DataType mismatch");
      } catch (IllegalArgumentException e) {
        // The expected exception.
      }

      try {
        matrix.copyTo(new int[2][3]);
        fail("should have failed on shape mismatch");
      } catch (IllegalArgumentException e) {
        // The expected exception.
      }
    }
  }

  @Test
  public void failCopyToOnScalar() {
    try (final Tensor scalar = Tensor.create(3)) {
      try {
        scalar.copyTo(3);
        fail("copyTo should fail on scalar tensors, suggesting use of primitive accessors instead");
      } catch (IllegalArgumentException e) {
        // The expected exception.
      }
    }
  }

  @Test
  public void failOnArbitraryObject() {
    try (Tensor t = Tensor.create(new Object())) {
      fail("should fail on creating a Tensor with a Java object that has not equivalent DataType");
    } catch (IllegalArgumentException e) {
      // The expected exception.
    }
  }

  @Test
  public void failOnZeroDimension() {
    try (Tensor t = Tensor.create(new int[3][0][1])) {
      fail("should fail on creating a Tensor where one of the dimensions is 0");
    } catch (IllegalArgumentException e) {
      // The expected exception.
    }
  }

  @Test
  public void useAfterClose() {
    int n = 4;
    Tensor t = Tensor.create(n);
    t.close();
    try {
      t.intValue();
    } catch (NullPointerException e) {
      // The expected exception.
    }
  }

  @Test
  public void fromHandle() {
    // fromHandle is a package-visible method intended for use when the C TF_Tensor object has been
    // created indepdently of the Java code. In practice, two Tensor instances MUST NOT have the
    // same native handle.
    //
    // An exception is made for this test, where the pitfalls of this is avoided by not calling
    // close() on both Tensors.
    final float[][] matrix = {{1, 2, 3}, {4, 5, 6}};
    try (Tensor src = Tensor.create(matrix)) {
      Tensor cpy = Tensor.fromHandle(src.getNativeHandle());
      assertEquals(src.dataType(), cpy.dataType());
      assertEquals(src.numDimensions(), cpy.numDimensions());
      assertArrayEquals(src.shape(), cpy.shape());
      assertArrayEquals(matrix, cpy.copyTo(new float[2][3]));
    }
  }
}
