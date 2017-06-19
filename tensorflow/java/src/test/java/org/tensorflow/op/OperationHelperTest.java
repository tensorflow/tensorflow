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

import java.util.List;

import org.junit.Test;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.TestUtil;

public class OperationHelperTest {

  @Test
  public void buildOperationAndCollectOutputs() {
    try (Graph g = new Graph()) {
      Scope s = new Scope(g);
      Output values = TestUtil.constant(g, "values", new int[] {0, 0, 1, 2, 2});

      OperationHelper unique = OperationHelper.create(s, "Unique");
      unique.builder().addInput(values);

      Output y = unique.nextOutput();
      Output idx = unique.nextOutput();
      assertEquals(1, y.shape().numDimensions());
      assertEquals(1, idx.shape().numDimensions());
    }
  }

  @Test
  public void buildOperationAndCollectOutputList() {
    try (Graph g = new Graph()) {
      Scope s = new Scope(g);
      Output splitDim = TestUtil.constant(g, "split_dim", 0);
      Output values = TestUtil.constant(g, "values", new int[] {0, 1, 2});

      OperationHelper split = OperationHelper.create(s, "Split");
      split.builder().addInput(splitDim).addInput(values).setAttr("num_split", 3);

      List<Output> outputs = split.nextOutputList("output");
      assertEquals(3, outputs.size());
      assertEquals(1, outputs.get(0).shape().numDimensions());
      assertEquals(1, outputs.get(0).shape().size(0));
    }
  }
}
