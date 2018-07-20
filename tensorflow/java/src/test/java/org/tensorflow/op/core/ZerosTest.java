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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.types.UInt8;

@RunWith(JUnit4.class)
public class ZerosTest {
  private static final float EPSILON = 1e-7f;
  
  @Test
  public void createIntZeros() {
    Shape shape = Shape.make(2, 2);
    int[] expected = new int[shape.numElements()]; // all zeros

    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Zeros<Integer> op = Zeros.create(scope, Integer.class, Shape.make(2, 2));
      Tensor<Integer> result = sess.runner().fetch(op.asOutput()).run().get(0).expect(Integer.class);
      int[] actual = new int[result.numElements()];
      result.copyTo(actual);
      assertArrayEquals(expected, actual);
    }
  }

  @Test
  public void createFloatZeros() {
    Shape shape = Shape.make(2, 2);
    float[] expected = new float[shape.numElements()]; // all zeros

    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Zeros<Float> op = Zeros.create(scope, Float.class, Shape.make(2, 2));
      Tensor<Float> result = sess.runner().fetch(op.asOutput()).run().get(0).expect(Float.class);
      float[] actual = new float[shape.numElements()];
      result.copyTo(actual);
      assertArrayEquals(expected, actual, EPSILON);
    }
  }

  @Test
  public void createDoubleZeros() {
    Shape shape = Shape.make(2, 2);
    double[] expected = new double[shape.numElements()]; // all zeros

    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Zeros<Double> op = Zeros.create(scope, Double.class, Shape.make(2, 2));
      Tensor<Double> result = sess.runner().fetch(op.asOutput()).run().get(0).expect(Double.class);
      double[] actual = new double[shape.numElements()];
      result.copyTo(actual);
      assertArrayEquals(expected, actual, EPSILON);
    }
  }

  @Test
  public void createLongZeros() {
    Shape shape = Shape.make(2, 2);
    float[] expected = new float[shape.numElements()]; // all zeros

    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Zeros<Long> op = Zeros.create(scope, Long.class, Shape.make(2, 2));
      Tensor<Long> result = sess.runner().fetch(op.asOutput()).run().get(0).expect(Long.class);
      float[] actual = new float[shape.numElements()];
      result.copyTo(actual);
      assertArrayEquals(expected, actual, 0.0f);
    }
  }

  @Test
  public void createBooleanZeros() {
    Shape shape = Shape.make(2, 2);
    boolean[] expected = new boolean[shape.numElements()]; // all zeros

    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Zeros<Boolean> op = Zeros.create(scope, Boolean.class, Shape.make(2, 2));
      Tensor<Boolean> result = sess.runner().fetch(op.asOutput()).run().get(0).expect(Boolean.class);
      boolean[] actual = new boolean[shape.numElements()];
      result.copyTo(actual);
      assertArrayEquals(expected, actual);
    }
  }

  @Test
  public void createUInt8Zeros() {
    Shape shape = Shape.make(2, 2);
    byte[] expected = new byte[shape.numElements()]; // all zeros

    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Zeros<UInt8> op = Zeros.create(scope, UInt8.class, Shape.make(2, 2));
      Tensor<UInt8> result = sess.runner().fetch(op.asOutput()).run().get(0).expect(UInt8.class);
      byte[] actual = new byte[shape.numElements()];
      result.copyTo(actual);
      assertArrayEquals(expected, actual);
    }
  }
  
  @Test(expected = IllegalArgumentException.class)
  public void cannotCreateStringZeros() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Zeros.create(scope, String.class, Shape.make(2, 2));
    }
  }

  @Test(expected = IllegalArgumentException.class)
  public void cannotCreateZerosWithUnknownDimensions() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Zeros.create(scope, Float.class, Shape.make(2, -1));
    }
  }
}
