/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.op;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

import java.util.HashSet;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.TestUtil;

/** Unit tests for {@link org.tensorflow.op.PrimitiveOp} */
@RunWith(JUnit4.class)
public class PrimitiveOpTest {

  @Test
  public void equalsHashcode() {
    try (Graph g = new Graph()) {
      Output array = TestUtil.constant(g, "array", new int[2]);

      PrimitiveOp test1 =
          new PrimitiveOp(g.opBuilder("Shape", "shape1").addInput(array).build()) {};
      PrimitiveOp test2 =
          new PrimitiveOp(g.opBuilder("Shape", "shape2").addInput(array).build()) {};
      PrimitiveOp test3 = new PrimitiveOp(test1.operation) {};

      // equals() tests
      assertNotEquals(test1, test2);
      assertEquals(test1, test3);
      assertEquals(test3, test1);
      assertNotEquals(test2, test3);

      // hashcode() tests
      Set<PrimitiveOp> ops = new HashSet<>();
      assertTrue(ops.add(test1));
      assertTrue(ops.add(test2));
      assertFalse(ops.add(test3));
    }
  }
}
