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
import static org.junit.Assert.assertEquals;
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
  public void createInt() {
    int value = 1;
    
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Constant<Integer> op = Constant.create(scope, value);
      try (Tensor<Integer> result = sess.runner().fetch(op).run().get(0).expect(Integer.class)) {
        assertEquals(value, result.intValue());
      }
    }
  }

  @Test
  public void createIntBuffer() {
    int[] ints = {1, 2, 3, 4};
    long[] shape = {4};

    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Constant<Integer> op = Constant.create(scope, shape, IntBuffer.wrap(ints));
      try (Tensor<?> result = sess.runner().fetch(op).run().get(0)) {
        int[] actual = new int[ints.length];
        assertArrayEquals(ints, result.expect(Integer.class).copyTo(actual));
      }
    }
  }

  @Test
  public void createFloat() {
    float value = 1;
    
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Constant<Float> op = Constant.create(scope, value);
      try (Tensor<?> result = sess.runner().fetch(op).run().get(0)) {
        assertEquals(value, result.expect(Float.class).floatValue(), 0.0f);
      }
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
      try (Tensor<?> result = sess.runner().fetch(op).run().get(0)) {
        float[] actual = new float[floats.length];
        assertArrayEquals(floats, result.expect(Float.class).copyTo(actual), EPSILON);
      }
    }
  }

  @Test
  public void createDouble() {
    double value = 1;
    
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Constant<Double> op = Constant.create(scope, value);
      try (Tensor<?> result = sess.runner().fetch(op).run().get(0)) {
        assertEquals(value, result.expect(Double.class).doubleValue(), 0.0);
      }
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
      try (Tensor<?> result = sess.runner().fetch(op).run().get(0)) {
        double[] actual = new double[doubles.length];
        assertArrayEquals(doubles, result.expect(Double.class).copyTo(actual), EPSILON);
      }
    }
  }

  @Test
  public void createLong() {
    long value = 1;
    
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Constant<Long> op = Constant.create(scope, value);
      try (Tensor<?> result = sess.runner().fetch(op).run().get(0)) {
        assertEquals(value, result.expect(Long.class).longValue());
      }
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
      try (Tensor<?> result = sess.runner().fetch(op).run().get(0)) {
        long[] actual = new long[longs.length];
        assertArrayEquals(longs, result.expect(Long.class).copyTo(actual));
      }
    }
  }

  @Test
  public void createBoolean() {
    boolean value = true;
    
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Constant<Boolean> op = Constant.create(scope, value);
      try (Tensor<?> result = sess.runner().fetch(op).run().get(0)) {
        assertEquals(value, result.expect(Boolean.class).booleanValue());
      }
    }
  }

  @Test
  public void createStringBuffer() throws IOException {
    byte[] data = {(byte) 1, (byte) 2, (byte) 3, (byte) 4};
    long[] shape = {};

    ByteArrayOutputStream baout = new ByteArrayOutputStream();
    DataOutputStream out = new DataOutputStream(baout);
    // We construct a TF_TString_Small tstring, which has the capacity for a 22 byte string.
    // The first 6 most significant bits of the first byte represent length; the remaining
    // 2-bits are type indicators, and are left as 0b00 to denote a TF_TSTR_SMALL type.
    assertTrue(data.length <= 22);
    out.writeByte(data.length << 2);
    out.write(data);
    out.close();
    byte[] content = baout.toByteArray();

    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Constant<String> op = Constant.create(scope, String.class, shape, ByteBuffer.wrap(content));
      try (Tensor<?> result = sess.runner().fetch(op).run().get(0)) {
        assertArrayEquals(data, result.expect(String.class).bytesValue());
      }
    }
  }
}
