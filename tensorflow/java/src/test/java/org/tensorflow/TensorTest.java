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

import java.lang.reflect.Array;
import java.nio.*;
import java.util.Arrays;

/** Unit tests for {@link org.tensorflow.Tensor}. */
@RunWith(JUnit4.class)
public class TensorTest {
  private static final double EPSILON = 1e-7;
  private static final float EPSILON_F = 1e-7f;

  // reusable sample data, varying by type and dimension
  static final int scalar = 42;
  static final long[] scalar_shape = {};
  static final double[] vector = {1.414, 2.718, 3.1415};
  static final float[] vector_f = {1.414f, 2.718f, 3.1415f};
  static final long[] vector_shape = {3};
  static final int[][] matrix = {{1, 2, 3}, {4, 5, 6}};
  static final long[] matrix_shape = {2, 3};
  static final long[][][] threeD = {
    {{1}, {3}, {5}, {7}, {9}}, {{2}, {4}, {6}, {8}, {0}},
  };
  static final long[] threeD_shape = {2, 5, 1};
  static final boolean[][][][] fourD = {
    {{{false, false, false, true}, {false, false, true, false}}},
    {{{false, false, true, true}, {false, true, false, false}}},
    {{{false, true, false, true}, {false, true, true, false}}},
  };
  static final long[] fourD_shape = {3, 1, 2, 4};

  @Test
  public void createWithBuffer() {
    // validate creating a tensor using a direct byte buffer (in host order)
    {
      ByteBuffer buf = ByteBuffer.allocateDirect(Double.SIZE / Byte.SIZE * vector.length).order(ByteOrder.nativeOrder());
      buf.asDoubleBuffer().put(vector);
      try(Tensor t = Tensor.create(DataType.DOUBLE, vector_shape, buf)) {
        double[] actual = new double[3];
        assertArrayEquals(vector, t.copyTo(actual), EPSILON);
      }
    }

    // validate byte order conversion
    {
      DoubleBuffer buf = ByteBuffer.allocate(Double.SIZE / Byte.SIZE * vector.length)
          .order(ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN)
          .asDoubleBuffer()
          .put(vector);
      buf.flip();
      try(Tensor t = Tensor.create(DataType.DOUBLE, vector_shape, buf)) {
        double[] actual = new double[3];
        assertArrayEquals(vector, t.copyTo(actual), EPSILON);
      }
    }

    // validate creating a tensor using a typed buffer
    {
      try(Tensor t = Tensor.create(DataType.INT32, scalar_shape, IntBuffer.wrap(new int[] { scalar }))) {
        assertEquals(scalar, t.intValue());
      }
      try(Tensor t = Tensor.create(DataType.DOUBLE, vector_shape, DoubleBuffer.wrap(vector))) {
        double[] actual = new double[3];
        assertArrayEquals(vector, t.copyTo(actual), EPSILON);
      }
      try(Tensor t = Tensor.create(DataType.FLOAT, vector_shape, FloatBuffer.wrap(vector_f))) {
        float[] actual = new float[3];
        assertArrayEquals(vector_f, t.copyTo(actual), EPSILON_F);
      }
      int[] matrix_ = (int[]) TestUtil.flatten(matrix, Integer.TYPE);
      try(Tensor t = Tensor.create(DataType.INT32, matrix_shape, IntBuffer.wrap(matrix_))) {
        int[][] actual = new int[2][3];
        assertArrayEquals(matrix, t.copyTo(actual));
      }
      long[] threeD_ = (long[]) TestUtil.flatten(threeD, Long.TYPE);
      try(Tensor t = Tensor.create(DataType.INT64, threeD_shape, LongBuffer.wrap(threeD_))) {
        long[][][] actual = new long[2][5][1];
        assertArrayEquals(threeD, t.copyTo(actual));
      }
      byte[] fourD_ = TestUtil.bool2byte((boolean[]) TestUtil.flatten(fourD, Boolean.TYPE));
      try(Tensor t = Tensor.create(DataType.BOOL, fourD_shape, ByteBuffer.wrap(fourD_))) {
        boolean[][][][] actual = new boolean[3][1][2][4];
        assertArrayEquals(fourD, t.copyTo(actual));
      }
    }

    // validate that incompatible buffers are rejected
    {
      try {
        Tensor.create(DataType.FLOAT, vector_shape, DoubleBuffer.wrap(vector));
        fail("should have failed on incompatible buffer");
      }
      catch(IllegalArgumentException e) {
      }
      try {
        Tensor.create(DataType.DOUBLE, vector_shape, FloatBuffer.wrap(vector_f));
        fail("should have failed on incompatible buffer");
      }
      catch(IllegalArgumentException e) {
      }
      int[] matrix_ = (int[]) TestUtil.flatten(matrix, Integer.TYPE);
      try {
        Tensor.create(DataType.FLOAT, matrix_shape, IntBuffer.wrap(matrix_));
        fail("should have failed on incompatible buffer");
      }
      catch(IllegalArgumentException e) {
      }
      long[] threeD_ = (long[]) TestUtil.flatten(threeD, Long.TYPE);
      try {
        Tensor.create(DataType.FLOAT, threeD_shape, LongBuffer.wrap(threeD_));
        fail("should have failed on incompatible buffer");
      }
      catch(IllegalArgumentException e) {
      }
      try {
        Tensor.create(DataType.FLOAT, new long[] { 1 }, ShortBuffer.wrap(new short[] { 1 }));
        fail("should have failed on incompatible buffer");
      }
      catch(IllegalArgumentException e) {
      }
    }
  }

  @Test
  public void readData() {

    Tensor tscalar = Tensor.create(scalar);
    Tensor tvector = Tensor.create(vector);
    Tensor tvector_f = Tensor.create(vector_f);
    Tensor tmatrix = Tensor.create(matrix);
    Tensor tthreeD = Tensor.create(threeD);
    Tensor tfourD = Tensor.create(fourD);
    try {

      // validate that any datatype is readable with ByteBuffer (content, position)
      {
        ByteBuffer bbuf = ByteBuffer.allocate(1024).order(ByteOrder.nativeOrder());

        bbuf.clear(); // FLOAT
        tvector_f.readData(bbuf);
        assertEquals(12, bbuf.position());
        bbuf.flip();
        assertEquals(vector_f[0], bbuf.asFloatBuffer().get(0), EPSILON);
        bbuf.clear(); // DOUBLE
        tvector.readData(bbuf);
        assertEquals(24, bbuf.position());
        bbuf.flip();
        assertEquals(vector[0], bbuf.asDoubleBuffer().get(0), EPSILON);
        bbuf.clear(); // INT32
        tmatrix.readData(bbuf);
        assertEquals(24, bbuf.position());
        bbuf.flip();
        assertEquals(matrix[0][0], bbuf.asIntBuffer().get(0));
        bbuf.clear(); // INT64
        tthreeD.readData(bbuf);
        assertEquals(80, bbuf.position());
        bbuf.flip();
        assertEquals(threeD[0][0][0], bbuf.asLongBuffer().get(0));
        bbuf.clear(); // (BOOL)
        tfourD.readData(bbuf);
        assertEquals(24, bbuf.position());
        bbuf.flip();
        assertEquals(fourD[0][0][0][0], bbuf.get(0) != 0);
        assertEquals(fourD[0][0][0][3], bbuf.get(3) != 0);
      }

      // validate the use of direct buffers
      {
        DoubleBuffer buf = ByteBuffer.allocateDirect(tvector.getDataByteSize())
            .order(ByteOrder.nativeOrder()).asDoubleBuffer();
        tvector.readData(buf);
        assertTrue(buf.isDirect());
        assertEquals(3, buf.position());
        assertEquals(vector[0], buf.get(0), EPSILON);
      }

      // validate typed buffers (content, position)
      {
        FloatBuffer buf = FloatBuffer.allocate(tvector_f.getDataByteSize() / (Float.SIZE / Byte.SIZE));
        tvector_f.readData(buf);
        assertEquals(3, buf.position());
        assertEquals(vector_f[0], buf.get(0), EPSILON);
      }
      {
        DoubleBuffer buf = DoubleBuffer.allocate(tvector.getDataByteSize() / (Double.SIZE / Byte.SIZE));
        tvector.readData(buf);
        assertEquals(3, buf.position());
        assertEquals(vector[0], buf.get(0), EPSILON);
      }
      {
        IntBuffer buf = IntBuffer.allocate(tmatrix.getDataByteSize() / (Integer.SIZE / Byte.SIZE));
        tmatrix.readData(buf);
        assertEquals(6, buf.position());
        assertEquals(matrix[0][0], buf.get(0));
      }
      {
        LongBuffer buf = LongBuffer.allocate(tthreeD.getDataByteSize() / (Long.SIZE / Byte.SIZE));
        tthreeD.readData(buf);
        assertEquals(10, buf.position());
        assertEquals(threeD[0][0][0], buf.get(0));
      }

      // validate byte order conversion
      {
        DoubleBuffer foreignBuf = ByteBuffer.allocate(tvector.getDataByteSize())
                .order(ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN)
                .asDoubleBuffer();
        tvector.readData(foreignBuf);
        foreignBuf.flip();
        double[] foreignBufData = new double[foreignBuf.remaining()];
        foreignBuf.get(foreignBufData);
        assertTrue(Arrays.equals(vector, foreignBufData));
      }

      // validate that incompatible buffers are rejected
      {
        IntBuffer badbuf1 = IntBuffer.allocate(128);
        try {
          tvector.readData(badbuf1);
          fail("should have failed on incompatible buffer");
        } catch (IllegalArgumentException e) {
        }
        FloatBuffer badbuf2 = FloatBuffer.allocate(128);
        try {
          tvector.readData(badbuf2);
          fail("should have failed on incompatible buffer");
        } catch (IllegalArgumentException e) {
        }
        DoubleBuffer badbuf3 = DoubleBuffer.allocate(128);
        try {
          tmatrix.readData(badbuf3);
          fail("should have failed on incompatible buffer");
        } catch (IllegalArgumentException e) {
        }
        LongBuffer badbuf4 = LongBuffer.allocate(128);
        try {
          tvector.readData(badbuf4);
          fail("should have failed on incompatible buffer");
        } catch (IllegalArgumentException e) {
        }
        ShortBuffer badbuf5 = ShortBuffer.allocate(128);
        try {
          tvector.readData(badbuf5);
          fail("should have failed on incompatible buffer");
        } catch (IllegalArgumentException e) {
        }
      }
    }
    finally {
      tscalar.close();
      tvector.close();
      tvector_f.close();
      tmatrix.close();
      tthreeD.close();
      tfourD.close();
    }
  }

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
    try (Tensor t = Tensor.create(vector)) {
      assertEquals(DataType.DOUBLE, t.dataType());
      assertEquals(1, t.numDimensions());
      assertArrayEquals(vector_shape, t.shape());

      double[] got = new double[3];
      assertArrayEquals(vector, t.copyTo(got), 0);
    }

    try (Tensor t = Tensor.create(matrix)) {
      assertEquals(DataType.INT32, t.dataType());
      assertEquals(2, t.numDimensions());
      assertArrayEquals(matrix_shape, t.shape());

      int[][] got = new int[2][3];
      assertArrayEquals(matrix, t.copyTo(got));
    }

    try (Tensor t = Tensor.create(threeD)) {
      assertEquals(DataType.INT64, t.dataType());
      assertEquals(3, t.numDimensions());
      assertArrayEquals(threeD_shape, t.shape());

      long[][][] got = new long[2][5][1];
      assertArrayEquals(threeD, t.copyTo(got));
    }

    try (Tensor t = Tensor.create(fourD)) {
      assertEquals(DataType.BOOL, t.dataType());
      assertEquals(4, t.numDimensions());
      assertArrayEquals(fourD_shape, t.shape());

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
