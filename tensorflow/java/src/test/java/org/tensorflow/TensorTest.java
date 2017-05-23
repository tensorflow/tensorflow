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

import static org.hamcrest.core.Is.is;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertThat;
import static org.junit.Assert.fail;
import static org.tensorflow.TensorMatcher.hasDataType;
import static org.tensorflow.TensorMatcher.scalar;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import org.hamcrest.TypeSafeMatcher;
import org.hamcrest.Description;
import org.junit.Rule;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.Test;

/** Unit tests for {@link org.tensorflow.Tensor}. */
@RunWith(JUnit4.class)
public class TensorTest {
  private static final double EPSILON = 1e-7;
  private static final float EPSILON_F = 1e-7f;

  @Rule
  public ExpectedException expectedException = ExpectedException.none();

  @Test
  public void createWithByteBuffer() {
    double[] doubles = {1d, 2d, 3d, 4d};
    long[] doubles_shape = {4};
    boolean[] bools = {true, false, true, false};
    long[] bools_shape = {4};
    byte[] bools_ = TestUtil.bool2byte(bools);
    byte[] strings = "test".getBytes();
    long[] strings_shape = {};
    byte[] strings_; // raw TF_STRING
    try (Tensor t = Tensor.create(strings)) {
      ByteBuffer to = ByteBuffer.allocate(t.numBytes());
      t.writeTo(to);
      strings_ = to.array();
    }

    // validate creating a tensor using a byte buffer
    {
      try (Tensor t = Tensor.create(DataType.BOOL, bools_shape, ByteBuffer.wrap(bools_))) {
        boolean[] actual = t.copyTo(new boolean[bools_.length]);
        for (int i = 0; i < bools.length; ++i) {
          assertEquals("" + i, bools[i], actual[i]);
        }
      }

      // note: the buffer is expected to contain raw TF_STRING (as per C API)
      try (Tensor t = Tensor.create(DataType.STRING, strings_shape, ByteBuffer.wrap(strings_))) {
        assertArrayEquals(strings, t.bytesValue());
      }
    }

    // validate creating a tensor using a direct byte buffer (in host order)
    {
      ByteBuffer buf = ByteBuffer.allocateDirect(8 * doubles.length).order(ByteOrder.nativeOrder());
      buf.asDoubleBuffer().put(doubles);
      try (Tensor t = Tensor.create(DataType.DOUBLE, doubles_shape, buf)) {
        double[] actual = new double[doubles.length];
        assertArrayEquals(doubles, t.copyTo(actual), EPSILON);
      }
    }

    // validate shape checking
    try (Tensor t =
        Tensor.create(DataType.BOOL, new long[bools_.length * 2], ByteBuffer.wrap(bools_))) {
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
    buf.flip();
    try (Tensor t = Tensor.create(new long[] {doubles.length}, buf)) {
      double[] actual = new double[doubles.length];
      assertArrayEquals(doubles, t.copyTo(actual), EPSILON);
    }
  }

  @Test
  public void whenCreateUsingDoubleBufferThenExpectedMatrixIsStoredInTensor() {
    final double[] expected = {1d, 2d, 3d, 4d};
    final long[] shape = {4};

    try (final Tensor t = Tensor.create(shape, DoubleBuffer.wrap(expected))) {
      final double[] actual = new double[expected.length];
      t.copyTo(actual);
      assertArrayEquals(expected, actual, EPSILON);
    }
  }

  @Test
  public void whenCreateUsingFloatBufferThenExpectedMatrixIsStoredInTensor() {
    final float[] expected = {1f, 2f, 3f, 4f};
    final long[] shape = {4};

    try (Tensor t = Tensor.create(shape, FloatBuffer.wrap(expected))) {
      final float[] actual = new float[expected.length];
      t.copyTo(actual);
      assertArrayEquals(expected, actual, EPSILON_F);
    }
  }
  
  @Test
  public void whenCreateUsingIntBufferThenExpectedMatrixIsStoredInTensor() {
    final int[] expected = {1, 2, 3, 4};
    final long[] shape = {4};

    try (Tensor t = Tensor.create(shape, IntBuffer.wrap(expected))) {
      final int[] actual = new int[expected.length];
      t.copyTo(actual);
      assertArrayEquals(expected, actual);
    }
  }

  @Test
  public void whenCreateUsingLongBufferThenExpectedMatrixIsStoredInTensor() {
    final long[] expected = {1L, 2L, 3L, 4L};
    final long[] shape = {4};

    try (Tensor t = Tensor.create(shape, LongBuffer.wrap(expected))) {
      final long[] actual = new long[expected.length];
      t.copyTo(actual);
      assertArrayEquals(expected, actual);
    }
  }

  @Test
  public void whenCreateUsingDoubleBufferWithIncorrectShapeThenThrowExpectedException() {
    expectedException.expect(IllegalArgumentException.class);
    expectedException.expectMessage("buffer with 4 elements is not compatible with a Tensor with shape [0, 0, 0, 0, 0]");
    final double[] expected = {1D, 2D, 3D, 4D};

    try(Tensor t = Tensor.create(new long[expected.length + 1], DoubleBuffer.wrap(expected))) {
    }
  }

  @Test
  public void whenCreateUsingFloatBufferWithIncorrectShapeThenThrowExpectedException() {
    expectedException.expect(IllegalArgumentException.class);
    expectedException.expectMessage("buffer with 4 elements is not compatible with a Tensor with shape [0, 0, 0, 0, 0]");
    final float[] expected = {1F, 2F, 3F, 4F};

    try(Tensor t = Tensor.create(new long[expected.length + 1], FloatBuffer.wrap(expected))) {
    }
  }

  @Test
  public void whenCreateUsingIntBufferWithIncorrectShapeThenThrowExpectedException() {
    expectedException.expect(IllegalArgumentException.class);
    expectedException.expectMessage("buffer with 4 elements is not compatible with a Tensor with shape [0, 0, 0, 0, 0]");
    final int[] expected = {1, 2, 3, 4};

    try(Tensor t = Tensor.create(new long[expected.length + 1], IntBuffer.wrap(expected))) {
    }
  }

  @Test
  public void whenCreateUsingLongBufferWithIncorrectShapeThenThrowExpectedException() {
    expectedException.expect(IllegalArgumentException.class);
    expectedException.expectMessage("buffer with 4 elements is not compatible with a Tensor with shape [0, 0, 0, 0, 0]");
    final long[] expected = {1L, 2L, 3L, 4L};

    try(Tensor t = Tensor.create(new long[expected.length + 1], LongBuffer.wrap(expected))) {
    }
  }

  @Test
  public void writeTo() {
    int[] ints = {1, 2, 3};
    float[] floats = {1f, 2f, 3f};
    double[] doubles = {1d, 2d, 3d};
    long[] longs = {1L, 2L, 3L};
    boolean[] bools = {true, false, true};

    try (Tensor tints = Tensor.create(ints);
        Tensor tfloats = Tensor.create(floats);
        Tensor tdoubles = Tensor.create(doubles);
        Tensor tlongs = Tensor.create(longs);
        Tensor tbools = Tensor.create(bools)) {

      // validate that any datatype is readable with ByteBuffer (content, position)
      {
        ByteBuffer bbuf = ByteBuffer.allocate(1024).order(ByteOrder.nativeOrder());

        bbuf.clear(); // FLOAT
        tfloats.writeTo(bbuf);
        assertEquals(tfloats.numBytes(), bbuf.position());
        bbuf.flip();
        assertEquals(floats[0], bbuf.asFloatBuffer().get(0), EPSILON);
        bbuf.clear(); // DOUBLE
        tdoubles.writeTo(bbuf);
        assertEquals(tdoubles.numBytes(), bbuf.position());
        bbuf.flip();
        assertEquals(doubles[0], bbuf.asDoubleBuffer().get(0), EPSILON);
        bbuf.clear(); // INT32
        tints.writeTo(bbuf);
        assertEquals(tints.numBytes(), bbuf.position());
        bbuf.flip();
        assertEquals(ints[0], bbuf.asIntBuffer().get(0));
        bbuf.clear(); // INT64
        tlongs.writeTo(bbuf);
        assertEquals(tlongs.numBytes(), bbuf.position());
        bbuf.flip();
        assertEquals(longs[0], bbuf.asLongBuffer().get(0));
        bbuf.clear(); // BOOL
        tbools.writeTo(bbuf);
        assertEquals(tbools.numBytes(), bbuf.position());
        bbuf.flip();
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
        foreignBuf.flip();
        double[] actual = new double[foreignBuf.remaining()];
        foreignBuf.get(actual);
        assertArrayEquals(doubles, actual, EPSILON);
      }

    }
  }

  @Test
  public void whenWriteToIncompatibleDoubleBufferThenThrowExpectedException() {
    expectedException.expect(IllegalArgumentException.class);
    expectedException.expectMessage("cannot use java.nio.HeapDoubleBuffer with Tensor of type BOOL");
    final boolean[] bools = {true, false, true};

    try (Tensor tensor = Tensor.create(bools)) {
      final DoubleBuffer badBuffer = DoubleBuffer.allocate(128);
      tensor.writeTo(badBuffer);
    }
  }

  @Test
  public void whenWriteToIncompatibleFloatBufferThenThrowExpectedException() {
    expectedException.expect(IllegalArgumentException.class);
    expectedException.expectMessage("cannot use java.nio.HeapFloatBuffer with Tensor of type BOOL");
    final boolean[] bools = {true, false, true};

    try (Tensor tensor = Tensor.create(bools)) {
      final FloatBuffer badBuffer = FloatBuffer.allocate(128);
      tensor.writeTo(badBuffer);
    }
  }

  @Test
  public void whenWriteToIncompatibleIntBufferThenThrowExpectedException() {
    expectedException.expect(IllegalArgumentException.class);
    expectedException.expectMessage("cannot use java.nio.HeapIntBuffer with Tensor of type BOOL");
    final boolean[] bools = {true, false, true};

    try (Tensor tensor = Tensor.create(bools)) {
      final IntBuffer badBuffer = IntBuffer.allocate(128);
      tensor.writeTo(badBuffer);
    } 
  }

  @Test
  public void whenWriteToIncompatibleLongBufferThenThrowExpectedException() {
    expectedException.expect(IllegalArgumentException.class);
    expectedException.expectMessage("cannot use java.nio.HeapLongBuffer with Tensor of type BOOL");
    final boolean[] bools = {true, false, true};

    try (Tensor tensor = Tensor.create(bools)) {
      final LongBuffer badBuffer = LongBuffer.allocate(128);
      tensor.writeTo(badBuffer);
    }
  }

  @Test
  public void whenCreateUsingDoubleThenReturnDoubleScalarWithExpectedValue() {
    final double expected = 3.1415D;
    try (Tensor t = Tensor.create(expected)) {
      assertThat(t, hasDataType(DataType.DOUBLE));
      assertThat(t, is(scalar()));
      assertEquals(expected, t.doubleValue(), EPSILON);
    }
  }

  @Test
  public void whenCreateUsingFloatThenReturnFloatScalarWithExpectedValue() {
    final float expected = 2.718F;
    try (Tensor t = Tensor.create(expected)) {
      assertThat(t, hasDataType(DataType.FLOAT));
      assertThat(t, is(scalar()));
      assertEquals(expected, t.floatValue(), EPSILON_F);
    }
  }

  @Test
  public void whenCreateUsingIntegerThenReturnIntegerScalarWithExpectedValue() {
    final int expected = -33;
    try (Tensor t = Tensor.create(expected)) {
      assertThat(t, hasDataType(DataType.INT32));
      assertThat(t, is(scalar()));
      assertThat(t.intValue(), is(expected));
    }
  }

  @Test
  public void whenCreateUsingLongThenReturnLongScalarWithExpectedValue() {
    final long expected = 8589934592L;
    try (Tensor t = Tensor.create(expected)) {
      assertThat(t, hasDataType(DataType.INT64));
      assertThat(t, is(scalar()));
      assertThat(t.longValue(), is(expected));
    } 
  }

  @Test
  public void whenCreateUsingBooleanThenReturnBooleanScalarWithExpectedValue() {
    final boolean expected = true;
    try (Tensor t = Tensor.create(expected)) {
      assertThat(t, hasDataType(DataType.BOOL));
      assertThat(t, is(scalar()));
      assertThat(t.booleanValue(), is(true));
    }
  }


  @Test
  public void whenCreateUsingByteArrayThenReturnStringScalarWithExpectedValue() {
    final byte[] bytes = {1, 2, 3, 4};
    try (Tensor t = Tensor.create(bytes)) {
      assertThat(t, hasDataType(DataType.STRING));
      assertThat(t, is(scalar()));
      assertArrayEquals(bytes, t.bytesValue());
    }
  }

  @Test
  public void nDimensional() {
    double[] vector = {1.414, 2.718, 3.1415};
    try (Tensor t = Tensor.create(vector)) {
      assertThat(t, hasDataType(DataType.DOUBLE));
      assertEquals(1, t.numDimensions());
      assertArrayEquals(new long[] {3}, t.shape());

      double[] got = new double[3];
      assertArrayEquals(vector, t.copyTo(got), EPSILON);
    }

    int[][] matrix = {{1, 2, 3}, {4, 5, 6}};
    try (Tensor t = Tensor.create(matrix)) {
      assertThat(t, hasDataType(DataType.INT32));
      assertEquals(2, t.numDimensions());
      assertArrayEquals(new long[] {2, 3}, t.shape());

      int[][] got = new int[2][3];
      assertArrayEquals(matrix, t.copyTo(got));
    }

    long[][][] threeD = {
      {{1}, {3}, {5}, {7}, {9}}, {{2}, {4}, {6}, {8}, {0}},
    };
    try (Tensor t = Tensor.create(threeD)) {
      assertThat(t, hasDataType(DataType.INT64));
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
      assertThat(t, hasDataType(DataType.BOOL));
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
  public void whenCopyToWithDimensionMismatchThenThrowExpectedException() {
    expectedException.expect(IllegalArgumentException.class);
    expectedException.expectMessage("cannot copy Tensor with 2 dimensions into an object with 1");

    try (final Tensor matrix = Tensor.create(new int[][] {{1, 2}, {3, 4}})) {
      matrix.copyTo(new int[2]);
    }
  }

  @Test
  public void whenCopyToDifferentDataTypePrimitiveThenThrowExpectedException() {
    expectedException.expect(IllegalArgumentException.class);
    expectedException.expectMessage("cannot copy Tensor with DataType INT32 into an object of type float[][]");

    try (final Tensor matrix = Tensor.create(new int[][] {{1, 2}, {3, 4}})) {
      matrix.copyTo(new float[2][2]);
    }
  }

  @Test
  public void whenCopyToMatrixWithDifferentShapeThenThrowExpectedException() {
    expectedException.expect(IllegalArgumentException.class);
    expectedException.expectMessage("cannot copy Tensor with shape [2, 2] into object with shape [2, 3]");

    try (final Tensor matrix = Tensor.create(new int[][] {{1, 2}, {3, 4}})) {
      matrix.copyTo(new int[2][3]);
    }
  }

  @Test
  public void whenCopyToOnScalarThenThrowExpectedException() {
    expectedException.expect(IllegalArgumentException.class);
    expectedException.expectMessage("copyTo() is not meant for scalar Tensors, use the scalar " +
        "accessor (floatValue(), intValue() etc.) instead");

    try (final Tensor scalar = Tensor.create(3)) {
        scalar.copyTo(3);
    }
  }

  @Test
  public void whenCreateTensorForObjectWithoutEquivalentDataTypeThenThrowExpectedException() {
    expectedException.expect(IllegalArgumentException.class);
    expectedException.expectMessage("cannot create Tensors of java.lang.Object");

    try(final Tensor t = Tensor.create(new Object())) {
    }
  }

  @Test
  public void whenCreatingTensorWithZeroDimensionThenThrowExpectedException() {
    expectedException.expect(IllegalArgumentException.class);
    expectedException.expectMessage("cannot create Tensors with a 0 dimension");

    try(final Tensor t = Tensor.create(new int[3][0][1])) {
    }
  }

  @Test
  public void whenUseTensorAfterCloseThenThrowExpectedException() {
    expectedException.expect(NullPointerException.class);
    // TODO change this message to make more sense to someone debugging code
    expectedException.expectMessage("close() was called on the Tensor");
    int n = 4;
    Tensor t = Tensor.create(n);
    t.close();

    t.intValue();
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
