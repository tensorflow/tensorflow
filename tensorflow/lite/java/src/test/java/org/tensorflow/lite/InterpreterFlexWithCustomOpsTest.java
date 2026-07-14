/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite;

import static com.google.common.truth.Truth.assertThat;

import java.nio.ByteBuffer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.lite.flex.FlexDelegate;

/**
 * Unit tests for {@link org.tensorflow.lite.Interpreter} that validate execution with models that
 * have user's defined TensorFlow ops.
 */
@RunWith(JUnit4.class)
public final class InterpreterFlexWithCustomOpsTest {

  private static final ByteBuffer DOUBLE_MODEL_BUFFER =
      TestUtils.getTestFileAsBuffer("tensorflow/lite/testdata/double_flex.bin");

  /** Smoke test validating that flex model with a user's defined TF op. */
  @Test
  public void testFlexModelWithUsersDefinedOp() throws Exception {
    try (Interpreter interpreter = new Interpreter(DOUBLE_MODEL_BUFFER)) {
      int[] oneD = {1, 2, 3, 4};
      int[][] twoD = {oneD};
      int[][] parsedOutputs = new int[1][4];
      interpreter.run(twoD, parsedOutputs);
      int[] outputOneD = parsedOutputs[0];
      int[] expected = {2, 4, 6, 8};
      assertThat(outputOneD).isEqualTo(expected);
    }
  }

  static {
    FlexDelegate.initTensorFlowForTesting();
  }
}
