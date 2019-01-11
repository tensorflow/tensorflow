/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;

@RunWith(JUnit4.class)
public final class GeneratedOperationsTest {

  @Test
  public void tensorInputTensorOutput() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Ops ops = Ops.create(g);
      Operand<Integer> x = ops.math().add(ops.constant(1), ops.constant(2));
      try (Tensor<Integer> result = sess.runner().fetch(x).run().get(0).expect(Integer.class)) {
        assertEquals(3, result.intValue());
      }
    }
  }
}
