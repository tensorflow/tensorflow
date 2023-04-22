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

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.types.UInt8;

/** Unit tests for {@link org.tensorflow.Tensor}. */
@RunWith(JUnit4.class)
public class TensorTest {
  private static final double EPSILON = 1e-7;
  private static final float EPSILON_F = 1e-7f;

  @Test
  public void createWithByteBuffer() {
    double[] doubles = {1d, 2d, 3d, 4d};
    long[] doubles_shape = {4};
    boolean[] bools = {true, false, true, false};
    long[] bools_shape = {4};
    byte[] bools_ = TestUtil.bool2byte(bools);
    byte[] strings = "test".getBytes(UTF_8);
    long[] strings_shape = {};
    byte[] strings_; // raw TF_STRING
    try (Tensor<String> t = Tensors.create(strings)) {
      ByteBuffer to = ByteBuffer.allocate(t.numBytes());
      t.writeTo(to);
      strings_ = to.array();
    }

    // validate creating a tensor using a byte buffer
    {
      try (Tensor<Boolean> t = Tensor.create(Boolean.class, bools_shape, ByteBuffer.wrap(bools_))) {
        boolean[] actual = t.copyTo(new boolean[bools_.length]);
        for (int i = 0; i < bools.length; ++i) {
          assertEquals("" + i, bools[i], actual[i]);
        }
      }

      // note: the buffer is expected to contain raw TF_STRING (as per C API)
      try (Tensor<String> t =
          Tensor.create(String.class, strings_shape, ByteBuffer.wrap(strings_))) {
        assertArrayEquals(strings, t.bytesValue());
      }
    }

    // validate creating a tensor using a direct byte buffer (in host order)
    {
      ByteBuffer buf = ByteBuffer.allocateDirect(8 * doubles.length).order(ByteOrder.nativeOrder());
      buf.asDoubleBuffer().put(doubles);
      try (Tensor<Double> t = Tensor.create(Double.class, doubles_shape, buf)) {
        double[] actual = new double[doubles.length];
        assertArrayEquals(doubles, t.copyTo(actual), EPSILON);
      }
    }

    // validate shape checking
    try (Tensor<Boolean> t =
        Tensor.create(Boolean.class, new long[bools_.length * 2], ByteBuffer.wrap(bools_))) {
      fail("should have failed on incompatible buffer");
    } catch (IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void createFromBufferWithNonNativeByteOrder() {
    double[] doubles = {1d, 2d, 3d, 4d};
    DoubleBuffer buf =
        ByteBuffer.allocate(8 * doubles.length)
            .order(
                ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN
                    ? ByteOrder.BIG_ENDIAN
                    : ByteOrder.LITTLE_ENDIAN)
            .asDoubleBuffer()
            .put(doubles);
    flipBuffer(buf);
    try (Tensor<Double> t = Tensor.create(new long[] {doubles.length}, buf)) {
      double[] actual = new double[doubles.length];
      assertArrayEquals(doubles, t.copyTo(actual), EPSILON);
    }
  }

  @Test
  public void createWithTypedBuffer() {
    int[] ints = {1, 2, 3, 4};
    float[] floats = {1f, 2f, 3f, 4f};
    double[] doubles = {1d, 2d, 3d, 4d};
    long[] longs = {1L, 2L, 3L, 4L};
    long[] shape = {4};

    // validate creating a tensor using a typed buffer
    {
      try (Tensor<Double> t = Tensor.create(shape, DoubleBuffer.wrap(doubles))) {
        double[] actual = new double[doubles.length];
        assertArrayEquals(doubles, t.copyTo(actual), EPSILON);
      }
      try (Tensor<Float> t = Tensor.create(shape, FloatBuffer.wrap(floats))) {
        float[] actual = new float[floats.length];
        assertArrayEquals(floats, t.copyTo(actual), EPSILON_F);
      }
      try (Tensor<Integer> t = Tensor.create(shape, IntBuffer.wrap(ints))) {
        int[] actual = new int[ints.length];
        assertArrayEquals(ints, t.copyTo(actual));
      }
      try (Tensor<Long> t = Tensor.create(shape, LongBuffer.wrap(longs))) {
        long[] actual = new long[longs.length];
        assertArrayEquals(longs, t.copyTo(actual));
      }
    }

    // validate shape-checking
    {
      try (Tensor<Double> t =
          Tensor.create(new long[doubles.length + 1], DoubleBuffer.wrap(doubles))) {
        fail("should have failed on incompatible buffer");
      } catch (IllegalArgumentException e) {
        // expected
      }
      try (Tensor<Float> t = Tensor.create(new long[floats.length + 1], FloatBuffer.wrap(floats))) {
        fail("should have failed on incompatible buffer");
      } catch (IllegalArgumentException e) {
        // expected
      }
      try (Tensor<Integer> t = Tensor.create(new long[ints.length + 1], IntBuffer.wrap(ints))) {
        fail("should have failed on incompatible buffer");
      } catch (IllegalArgumentException e) {
        // expected
      }
      try (Tensor<Long> t = Tensor.create(new long[longs.length + 1], LongBuffer.wrap(longs))) {
        fail("should have failed on incompatible buffer");
      } catch (IllegalArgumentException e) {
        // expected
      }
    }
  }

  @Test
  public void writeTo() {
    int[] ints = {1, 2, 3};
    float[] floats = {1f, 2f, 3f};
    double[] doubles = {1d, 2d, 3d};
    long[] longs = {1L, 2L, 3L};
    boolean[] bools = {true, false, true};

    try (Tensor<Integer> tints = Tensors.create(ints);
        Tensor<Float> tfloats = Tensors.create(floats);
        Tensor<Double> tdoubles = Tensors.create(doubles);
        Tensor<Long> tlongs = Tensors.create(longs);
        Tensor<Boolean> tbools = Tensors.create(bools)) {

      // validate that any datatype is readable with ByteBuffer (content, position)
      {
        ByteBuffer bbuf = ByteBuffer.allocate(1024).order(ByteOrder.nativeOrder());

        clearBuffer(bbuf); // FLOAT
        tfloats.writeTo(bbuf);
        assertEquals(tfloats.numBytes(), bbuf.position());
        flipBuffer(bbuf);
        assertEquals(floats[0], bbuf.asFloatBuffer().get(0), EPSILON);
        clearBuffer(bbuf); // DOUBLE
        tdoubles.writeTo(bbuf);
        assertEquals(tdoubles.numBytes(), bbuf.position());
        flipBuffer(bbuf);
        assertEquals(doubles[0], bbuf.asDoubleBuffer().get(0), EPSILON);
        clearBuffer(bbuf); // INT32
        tints.writeTo(bbuf);
        assertEquals(tints.numBytes(), bbuf.position());
        flipBuffer(bbuf);
        assertEquals(ints[0], bbuf.asIntBuffer().get(0));
        clearBuffer(bbuf); // INT64
        tlongs.writeTo(bbuf);
        assertEquals(tlongs.numBytes(), bbuf.position());
        flipBuffer(bbuf);
        assertEquals(longs[0], bbuf.asLongBuffer().get(0));
        clearBuffer(bbuf); // BOOL
        tbools.writeTo(bbuf);
        assertEquals(tbools.numBytes(), bbuf.position());
        flipBuffer(bbuf);
        assertEquals(bools[0], bbuf.get(0) != 0);
      }

      // validate the use of direct buffers
      {
        DoubleBuffer buf =
            ByteBuffer.allocateDirect(tdoubles.numBytes())
                .order(ByteOrder.nativeOrder())
                .asDoubleBuffer();
        tdoubles.writeTo(buf);
        assertTrue(buf.isDirect());
        assertEquals(tdoubles.numElements(), buf.position());
        assertEquals(doubles[0], buf.get(0), EPSILON);
      }

      // validate typed buffers (content, position)
      {
        FloatBuffer buf = FloatBuffer.allocate(tfloats.numElements());
        tfloats.writeTo(buf);
        assertEquals(tfloats.numElements(), buf.position());
        assertEquals(floats[0], buf.get(0), EPSILON);
      }
      {
        DoubleBuffer buf = DoubleBuffer.allocate(tdoubles.numElements());
        tdoubles.writeTo(buf);
        assertEquals(tdoubles.numElements(), buf.position());
        assertEquals(doubles[0], buf.get(0), EPSILON);
      }
      {
        IntBuffer buf = IntBuffer.allocate(tints.numElements());
        tints.writeTo(buf);
        assertEquals(tints.numElements(), buf.position());
        assertEquals(ints[0], buf.get(0));
      }
      {
        LongBuffer buf = LongBuffer.allocate(tlongs.numElements());
        tlongs.writeTo(buf);
        assertEquals(tlongs.numElements(), buf.position());
        assertEquals(longs[0], buf.get(0));
      }

      // validate byte order conversion
      {
        DoubleBuffer foreignBuf =
            ByteBuffer.allocate(tdoubles.numBytes())
                .order(
                    ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN
                        ? ByteOrder.BIG_ENDIAN
                        : ByteOrder.LITTLE_ENDIAN)
                .asDoubleBuffer();
        tdoubles.writeTo(foreignBuf);
        flipBuffer(foreignBuf);
        double[] actual = new double[foreignBuf.remaining()];
        foreignBuf.get(actual);
        assertArrayEquals(doubles, actual, EPSILON);
      }

      // validate that incompatible buffers are rejected
      {
        IntBuffer badbuf1 = IntBuffer.allocate(128);
        try {
          tbools.writeTo(badbuf1);
          fail("should have failed on incompatible buffer");
        } catch (IllegalArgumentException e) {
          // expected
        }
        FloatBuffer badbuf2 = FloatBuffer.allocate(128);
        try {
          tbools.writeTo(badbuf2);
          fail("should have failed on incompatible buffer");
        } catch (IllegalArgumentException e) {
          // expected
        }
        DoubleBuffer badbuf3 = DoubleBuffer.allocate(128);
        try {
          tbools.writeTo(badbuf3);
          fail("should have failed on incompatible buffer");
        } catch (IllegalArgumentException e) {
          // expected
        }
        LongBuffer badbuf4 = LongBuffer.allocate(128);
        try {
          tbools.writeTo(badbuf4);
          fail("should have failed on incompatible buffer");
        } catch (IllegalArgumentException e) {
          // expected
        }
      }
    }
  }

  @Test
  public void scalars() {
    try (Tensor<Float> t = Tensors.create(2.718f)) {
      assertEquals(DataType.FLOAT, t.dataType());
      assertEquals(0, t.numDimensions());
      assertEquals(0, t.shape().length);
      assertEquals(2.718f, t.floatValue(), EPSILON_F);
    }

    try (Tensor<Double> t = Tensors.create(3.1415)) {
      assertEquals(DataType.DOUBLE, t.dataType());
      assertEquals(0, t.numDimensions());
      assertEquals(0, t.shape().length);
      assertEquals(3.1415, t.doubleValue(), EPSILON);
    }

    try (Tensor<Integer> t = Tensors.create(-33)) {
      assertEquals(DataType.INT32, t.dataType());
      assertEquals(0, t.numDimensions());
      assertEquals(0, t.shape().length);
      assertEquals(-33, t.intValue());
    }

    try (Tensor<Long> t = Tensors.create(8589934592L)) {
      assertEquals(DataType.INT64, t.dataType());
      assertEquals(0, t.numDimensions());
      assertEquals(0, t.shape().length);
      assertEquals(8589934592L, t.longValue());
    }

    try (Tensor<Boolean> t = Tensors.create(true)) {
      assertEquals(DataType.BOOL, t.dataType());
      assertEquals(0, t.numDimensions());
      assertEquals(0, t.shape().length);
      assertTrue(t.booleanValue());
    }

    final byte[] bytes = {1, 2, 3, 4};
    try (Tensor<String> t = Tensors.create(bytes)) {
      assertEquals(DataType.STRING, t.dataType());
      assertEquals(0, t.numDimensions());
      assertEquals(0, t.shape().length);
      assertArrayEquals(bytes, t.bytesValue());
    }
  }

  @Test
  public void nDimensional() {
    double[] vector = {1.414, 2.718, 3.1415};
    try (Tensor<Double> t = Tensors.create(vector)) {
      assertEquals(DataType.DOUBLE, t.dataType());
      assertEquals(1, t.numDimensions());
      assertArrayEquals(new long[] {3}, t.shape());

      double[] got = new double[3];
      assertArrayEquals(vector, t.copyTo(got), EPSILON);
    }

    int[][] matrix = {{1, 2, 3}, {4, 5, 6}};
    try (Tensor<Integer> t = Tensors.create(matrix)) {
      assertEquals(DataType.INT32, t.dataType());
      assertEquals(2, t.numDimensions());
      assertArrayEquals(new long[] {2, 3}, t.shape());

      int[][] got = new int[2][3];
      assertArrayEquals(matrix, t.copyTo(got));
    }

    long[][][] threeD = {
      {{1}, {3}, {5}, {7}, {9}}, {{2}, {4}, {6}, {8}, {0}},
    };
    try (Tensor<Long> t = Tensors.create(threeD)) {
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
    try (Tensor<Boolean> t = Tensors.create(fourD)) {
      assertEquals(DataType.BOOL, t.dataType());
      assertEquals(4, t.numDimensions());
      assertArrayEquals(new long[] {3, 1, 2, 4}, t.shape());

      boolean[][][][] got = new boolean[3][1][2][4];
      assertArrayEquals(fourD, t.copyTo(got));
    }
  }

  @Test
  public void testNDimensionalStringTensor() {
    byte[][][] matrix = new byte[4][3][];
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 3; ++j) {
        matrix[i][j] = String.format("(%d, %d) = %d", i, j, i << j).getBytes(UTF_8);
      }
    }
    try (Tensor<String> t = Tensors.create(matrix)) {
      assertEquals(DataType.STRING, t.dataType());
      assertEquals(2, t.numDimensions());
      assertArrayEquals(new long[] {4, 3}, t.shape());

      byte[][][] got = t.copyTo(new byte[4][3][]);
      assertEquals(4, got.length);
      for (int i = 0; i < 4; ++i) {
        assertEquals(String.format("%d", i), 3, got[i].length);
        for (int j = 0; j < 3; ++j) {
          assertArrayEquals(String.format("(%d, %d)", i, j), matrix[i][j], got[i][j]);
        }
      }
    }
  }

  @Test
  public void testUInt8Tensor() {
    byte[] vector = new byte[] {1, 2, 3, 4};
    try (Tensor<UInt8> t = Tensor.create(vector, UInt8.class)) {
      assertEquals(DataType.UINT8, t.dataType());
      assertEquals(1, t.numDimensions());
      assertArrayEquals(new long[] {4}, t.shape());

      byte[] got = t.copyTo(new byte[4]);
      assertArrayEquals(vector, got);
    }
  }

  @Test
  public void testCreateFromArrayOfBoxed() {
    Integer[] vector = new Integer[] {1, 2, 3, 4};
    try (Tensor<Integer> t = Tensor.create(vector, Integer.class)) {
      fail("Tensor.create() should fail because it was given an array of boxed values");
    } catch (IllegalArgumentException e) {
      // The expected exception
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
    try (Tensor<?> t = Tensor.create(invalid)) {
      fail("Tensor.create() should fail because of differing sizes in the 3rd dimension");
    } catch (IllegalArgumentException e) {
      // The expected exception.
    }
  }

  @Test
  public void failCopyToOnIncompatibleDestination() {
    try (final Tensor<Integer> matrix = Tensors.create(new int[][] {{1, 2}, {3, 4}})) {
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
    try (final Tensor<Integer> scalar = Tensors.create(3)) {
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
    try (Tensor<?> t = Tensor.create(new Object())) {
      fail("should fail on creating a Tensor with a Java object that has no equivalent DataType");
    } catch (IllegalArgumentException e) {
      // The expected exception.
    }
  }

  @Test
  public void failOnZeroDimension() {
    try (Tensor<Integer> t = Tensors.create(new int[3][0][1])) {
      fail("should fail on creating a Tensor where one of the dimensions is 0");
    } catch (IllegalArgumentException e) {
      // The expected exception.
    }
  }

  @Test
  public void useAfterClose() {
    int n = 4;
    Tensor<?> t = Tensor.create(n);
    t.close();
    try {
      t.intValue();
    } catch (NullPointerException e) {
      // The expected exception.
    }
  }

  @Test
  public void eagerTensorIsReleasedAfterSessionIsClosed() {
    Tensor<Integer> sum;
    try (EagerSession session = EagerSession.create()) {
      Output<?> x = TestUtil.constant(session, "Const1", 10);
      Output<?> y = TestUtil.constant(session, "Const2", 20);
      sum = TestUtil.<Integer>addN(session, x, y).tensor();
      assertNotEquals(0L, sum.getNativeHandle());
      assertEquals(30, sum.intValue());
    }
    assertEquals(0L, sum.getNativeHandle());
    try {
      sum.intValue();
      fail();
    } catch (NullPointerException e) {
      // expected.
    }
  }

  @Test
  public void fromHandle() {
    // fromHandle is a package-visible method intended for use when the C TF_Tensor object has been
    // created independently of the Java code. In practice, two Tensor instances MUST NOT have the
    // same native handle.
    //
    // An exception is made for this test, where the pitfalls of this is avoided by not calling
    // close() on both Tensors.
    final float[][] matrix = {{1, 2, 3}, {4, 5, 6}};
    try (Tensor<Float> src = Tensors.create(matrix)) {
      Tensor<Float> cpy = Tensor.fromHandle(src.getNativeHandle()).expect(Float.class);
      assertEquals(src.dataType(), cpy.dataType());
      assertEquals(src.numDimensions(), cpy.numDimensions());
      assertArrayEquals(src.shape(), cpy.shape());
      assertArrayEquals(matrix, cpy.copyTo(new float[2][3]));
    }
  }

  @Test
  public void gracefullyFailCreationFromNullArrayForStringTensor() {
    // Motivated by: https://github.com/tensorflow/tensorflow/issues/17130
    byte[][] array = new byte[1][];
    try {
      Tensors.create(array);
    } catch (NullPointerException e) {
      // expected.
    }
  }

  // Workaround for cross compiliation
  // (e.g., javac -source 1.9 -target 1.8).
  //
  // In Java 8 and prior, subclasses of java.nio.Buffer (e.g., java.nio.DoubleBuffer) inherited the
  // "flip()" and "clear()" methods from java.nio.Buffer resulting in the signature:
  //   Buffer flip();
  // In Java 9 these subclasses had their own methods like:
  //   DoubleBuffer flip();
  // As a result, compiling for 1.9 source for a target of JDK 1.8 would result in errors at runtime
  // like:
  //
  // java.lang.NoSuchMethodError: java.nio.DoubleBuffer.flip()Ljava/nio/DoubleBuffer
  private static void flipBuffer(Buffer buf) {
    buf.flip();
  }

  // See comment for flipBuffer()
  private static void clearBuffer(Buffer buf) {
    buf.clear();
  }
}
