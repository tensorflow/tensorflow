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

package org.tensorflow.lite;

import static com.google.common.truth.Truth.assertThat;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link org.tensorflow.lite.Interpreter} that validate execution with models that
 * have TensorFlow ops.
 */
@RunWith(JUnit4.class)
public final class InterpreterFlexTest {

  private static final File FLEX_MODEL_FILE =
      new File("tensorflow/lite/testdata/multi_add_flex.bin");

  /** Smoke test validating that flex model loading works when the flex delegate is linked. */
  @Test
  public void testFlexModel() throws Exception {
    try (Interpreter interpreter = new Interpreter(FLEX_MODEL_FILE)) {
      assertThat(interpreter.getInputTensorCount()).isEqualTo(4);
      assertThat(interpreter.getInputTensor(0).dataType()).isEqualTo(DataType.FLOAT32);
      assertThat(interpreter.getInputTensor(1).dataType()).isEqualTo(DataType.FLOAT32);
      assertThat(interpreter.getInputTensor(2).dataType()).isEqualTo(DataType.FLOAT32);
      assertThat(interpreter.getInputTensor(3).dataType()).isEqualTo(DataType.FLOAT32);
      assertThat(interpreter.getOutputTensorCount()).isEqualTo(2);
      assertThat(interpreter.getOutputTensor(0).dataType()).isEqualTo(DataType.FLOAT32);
      assertThat(interpreter.getOutputTensor(1).dataType()).isEqualTo(DataType.FLOAT32);
      Object[] inputs = new Object[] {new float[1], new float[1], new float[1], new float[1]};
      Map<Integer, Object> outputs = new HashMap<>();
      outputs.put(0, new float[1]);
      outputs.put(1, new float[1]);
      interpreter.runForMultipleInputsOutputs(inputs, outputs);
    }
  }

  static {
    TensorFlowLite.initTensorFlow();
  }
}
