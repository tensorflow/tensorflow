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

package org.tensorflow;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.List;

/** Unit tests for {@link org.tensorflow.Operation}. */
@RunWith(JUnit4.class)
public class OperationTest {

  @Test
  public void outputListLength() {
    try (Graph g = new Graph()) {

      checkSplit(g, "t1", new int[] {0, 1}, 1);
      checkSplit(g, "t2", new int[] {0, 1}, 2);
      checkSplit(g, "t3", new int[] {0, 1, 2}, 3);
    }
  }

  private void checkSplit(Graph g, String name, int[] values, int num) {
    Operation op =
        g.opBuilder("Split", "split_op_" + name)
            .addInput(TestUtil.constant(g, "split_dim_" + name, 0))
            .addInput(TestUtil.constant(g, "values_" + name, values))
            .setAttr("num_split", num)
            .build();

    assertEquals(num, op.numOutputs());
    try {
      op.outputListLength("unknown_name");
      fail("Did not catch bad name");
    } catch (IllegalArgumentException iae) {
      // expected
    }
    assertEquals(num, op.outputListLength("output"));
  }
}
