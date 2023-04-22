/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.fail;

import java.util.List;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.types.UInt8;

@RunWith(JUnit4.class)
public class ZerosTest {
  private static final float EPSILON = 1e-7f;
  
  @Test
  public void createIntZeros() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      long[] shape = {2, 2};
      Zeros<Integer> op = Zeros.create(scope, Constant.create(scope, shape), Integer.class);
      try (Tensor<?> result = sess.runner().fetch(op).run().get(0)) {
        int[][] actual = result.expect(Integer.class).copyTo(new int[(int)shape[0]][(int)shape[1]]);
        for (int i = 0; i < actual.length; ++i) {
          for (int j = 0; j < actual[i].length; ++j) {
            assertEquals(0, actual[i][j]);
          }
        }
      }
    }
  }

  @Test
  public void createFloatZeros() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      long[] shape = {2, 2};
      Zeros<Float> op = Zeros.create(scope, Constant.create(scope, shape), Float.class);
      try (Tensor<?> result = sess.runner().fetch(op.asOutput()).run().get(0)) {
        float[][] actual = result.expect(Float.class).copyTo(new float[(int)shape[0]][(int)shape[1]]);
        for (int i = 0; i < actual.length; ++i) {
          for (int j = 0; j < actual[i].length; ++j) {
            assertEquals(0.0f, actual[i][j], EPSILON);
          }
        }
      }
    }
  }

  @Test
  public void createDoubleZeros() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      long[] shape = {2, 2};
      Zeros<Double> op = Zeros.create(scope, Constant.create(scope, shape), Double.class);
      try (Tensor<?> result = sess.runner().fetch(op.asOutput()).run().get(0)) {
        double[][] actual = result.expect(Double.class).copyTo(new double[(int)shape[0]][(int)shape[1]]);
        for (int i = 0; i < actual.length; ++i) {
          for (int j = 0; j < actual[i].length; ++j) {
            assertEquals(0.0, actual[i][j], EPSILON);
          }
        }
      }
    }
  }

  @Test
  public void createLongZeros() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      long[] shape = {2, 2};
      Zeros<Long> op = Zeros.create(scope, Constant.create(scope, shape), Long.class);
      try (Tensor<?> result = sess.runner().fetch(op.asOutput()).run().get(0)) {
        long[][] actual = result.expect(Long.class).copyTo(new long[(int)shape[0]][(int)shape[1]]);
        for (int i = 0; i < actual.length; ++i) {
          for (int j = 0; j < actual[i].length; ++j) {
            assertEquals(0L, actual[i][j]);
          }
        }
      }
    }
  }

  @Test
  public void createBooleanZeros() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      long[] shape = {2, 2};
      Zeros<Boolean> op = Zeros.create(scope, Constant.create(scope, shape), Boolean.class);
      try (Tensor<?> result = sess.runner().fetch(op.asOutput()).run().get(0)) {
        boolean[][] actual = result.expect(Boolean.class).copyTo(new boolean[(int)shape[0]][(int)shape[1]]);
        for (int i = 0; i < actual.length; ++i) {
          for (int j = 0; j < actual[i].length; ++j) {
            assertFalse(actual[i][j]);
          }
        }
      }
    }
  }

  @Test
  public void createUInt8Zeros() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      long[] shape = {2, 2};
      Zeros<UInt8> op = Zeros.create(scope, Constant.create(scope, shape), UInt8.class);
      try (Tensor<?> result = sess.runner().fetch(op.asOutput()).run().get(0)) {
        byte[][] actual = result.expect(UInt8.class).copyTo(new byte[(int)shape[0]][(int)shape[1]]);
        result.copyTo(actual);
        for (int i = 0; i < actual.length; ++i) {
          for (int j = 0; j < actual[i].length; ++j) {
            assertEquals(0, actual[i][j]);
          }
        }
      }
    }
  }
  
  @Test
  public void cannotCreateStringZeros() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      long[] shape = {2, 2};
      Zeros.create(scope, Constant.create(scope, shape), String.class);
      fail();
    } catch (IllegalArgumentException expected) {}
  }
  
  @Test
  public void operationsComposingZerosAreCorrectlyNamed() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      long[] shape = {2, 2};
      Zeros<Float> zeros = Zeros.create(scope.withSubScope("test"), Constant.create(scope, shape), Float.class);
      List<Tensor<?>> results = sess.runner().addTarget("test/Zeros/Zero").addTarget("test/Zeros/Fill").run();
    }
  }
}
