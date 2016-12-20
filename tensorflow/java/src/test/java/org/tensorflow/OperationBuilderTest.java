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

import static org.junit.Assert.fail;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.OperationBuilder}. */
@RunWith(JUnit4.class)
public class OperationBuilderTest {
  // TODO(ashankar): Restore this test once the C API gracefully handles mixing graphs and
  // operations instead of segfaulting.
  // @Test
  public void failWhenMixingOperationsOnDifferentGraphs() {
    try (Graph g1 = new Graph();
        Graph g2 = new Graph()) {
      Output c1 = TestUtil.constant(g1, "C1", 3);
      Output c2 = TestUtil.constant(g2, "C2", 3);
      TestUtil.addN(g1, c1, c1);
      try {
        TestUtil.addN(g2, c1, c2);
      } catch (Exception e) {
        fail(e.toString());
      }
    }
  }

  @Test
  public void failOnUseAfterBuild() {
    try (Graph g = new Graph();
        Tensor t = Tensor.create(1)) {
      OperationBuilder b =
          g.opBuilder("Const", "Const").setAttr("dtype", t.dataType()).setAttr("value", t);
      b.build();
      try {
        b.setAttr("dtype", t.dataType());
      } catch (IllegalStateException e) {
        // expected exception.
      }
    }
  }

  @Test
  public void failOnUseAfterGraphClose() {
    OperationBuilder b = null;
    try (Graph g = new Graph();
        Tensor t = Tensor.create(1)) {
      b = g.opBuilder("Const", "Const").setAttr("dtype", t.dataType()).setAttr("value", t);
    }
    try {
      b.build();
    } catch (IllegalStateException e) {
      // expected exception.
    }
  }
}
