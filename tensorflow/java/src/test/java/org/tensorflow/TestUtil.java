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

/** Static utility functions. */
public class TestUtil {
  public static Output constant(Graph g, String name, Object value) {
    try (Tensor t = Tensor.create(value)) {
      return g.opBuilder("Const", name)
          .setAttr("dtype", t.dataType())
          .setAttr("value", t)
          .build()
          .output(0);
    }
  }

  public static Output placeholder(Graph g, String name, DataType dtype) {
    return g.opBuilder("Placeholder", name).setAttr("dtype", dtype).build().output(0);
  }

  public static Output addN(Graph g, Output... inputs) {
    return g.opBuilder("AddN", "AddN").addInputList(inputs).build().output(0);
  }

  public static Output matmul(
      Graph g, String name, Output a, Output b, boolean transposeA, boolean transposeB) {
    return g.opBuilder("MatMul", name)
        .addInput(a)
        .addInput(b)
        .setAttr("transpose_a", transposeA)
        .setAttr("transpose_b", transposeB)
        .build()
        .output(0);
  }

  public static void transpose_A_times_X(Graph g, int[][] a) {
    matmul(g, "Y", constant(g, "A", a), placeholder(g, "X", DataType.INT32), true, false);
  }
}
