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
import static org.junit.Assert.assertSame;

import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Output;
import org.tensorflow.TestUtil;

/** Unit tests for {@link org.tensorflow.op.Operands}. */
@RunWith(JUnit4.class)
public class OperandsTest {

  @Test
  public void createOutputArrayFromOperandList() {
    try (Graph g = new Graph()) {
      Operation split = TestUtil.split(g, "split", new int[] {0, 1, 2}, 3);
      List<Output<Integer>> list =
          Arrays.asList(split.<Integer>output(0), split.<Integer>output(2));
      Output<?>[] array = Operands.asOutputs(list);
      assertEquals(list.size(), array.length);
      assertSame(array[0], list.get(0));
      assertSame(array[1], list.get(1));
    }
  }
}
