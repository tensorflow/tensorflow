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

package org.tensorflow.op.core;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;

@RunWith(JUnit4.class)
public class ConstantTest {
  private static final float EPSILON = 1e-7f;

  @Test
  public void createIntBuffer() {
    int[] ints = {1, 2, 3, 4};
    long[] shape = {4};

    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Constant<Integer> op = Constant.create(scope, shape, IntBuffer.wrap(ints));
      Tensor<Integer> result = sess.runner().fetch(op.asOutput())
          .run().get(0).expect(Integer.class);
      int[] actual = new int[ints.length];
      assertArrayEquals(ints, result.copyTo(actual));
    }
  }

  @Test
  public void createFloatBuffer() {
    float[] floats = {1, 2, 3, 4};
    long[] shape = {4};

    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Constant<Float> op = Constant.create(scope, shape, FloatBuffer.wrap(floats));
      Tensor<Float> result = sess.runner().fetch(op.asOutput()).run().get(0).expect(Float.class);
      float[] actual = new float[floats.length];
      assertArrayEquals(floats, result.copyTo(actual), EPSILON);
    }
  }

  @Test
  public void createDoubleBuffer() {
    double[] doubles = {1, 2, 3, 4};
    long[] shape = {4};

    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Constant<Double> op = Constant.create(scope, shape, DoubleBuffer.wrap(doubles));
      Tensor<Double> result = sess.runner().fetch(op.asOutput()).run().get(0).expect(Double.class);
      double[] actual = new double[doubles.length];
      assertArrayEquals(doubles, result.copyTo(actual), EPSILON);
    }
  }

  @Test
  public void createLongBuffer() {
    long[] longs = {1, 2, 3, 4};
    long[] shape = {4};

    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Constant<Long> op = Constant.create(scope, shape, LongBuffer.wrap(longs));
      Tensor<Long> result = sess.runner().fetch(op.asOutput()).run().get(0).expect(Long.class);
      long[] actual = new long[longs.length];
      assertArrayEquals(longs, result.copyTo(actual));
    }
  }

  @Test
  public void createStringBuffer() throws IOException {

    byte[] data = {(byte) 1, (byte) 2, (byte) 3, (byte) 4};
    long[] shape = {};

    // byte arrays (DataType.STRING in Tensorflow) are encoded as an offset in the data buffer,
    // followed by a varint encoded size, followed by the data.
    ByteArrayOutputStream baout = new ByteArrayOutputStream();
    DataOutputStream out = new DataOutputStream(baout);
    // Offset in array.
    out.writeLong(0L);
    // Varint encoded length of buffer.
    // For any number < 0x80, the varint encoding is simply the number itself.
    // https://developers.google.com/protocol-buffers/docs/encoding#varints
    assertTrue(data.length < 0x80);
    out.write(data.length);
    out.write(data);
    out.close();
    byte[] content = baout.toByteArray();

    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Constant<String> op = Constant.create(scope, String.class, shape, ByteBuffer.wrap(content));
      Tensor<String> result = sess.runner().fetch(op.asOutput()).run().get(0).expect(String.class);
      assertArrayEquals(data, result.bytesValue());
    }
  }
}
