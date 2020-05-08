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

package org.tensorflow.lite.gpu;

import static com.google.common.truth.Truth.assertThat;

import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.TestUtils;

/** Unit tests for {@link org.tensorflow.lite.gpu.GpuDelegate}. */
@RunWith(JUnit4.class)
public final class GpuDelegateTest {

  private static final String MODEL_PATH = "tensorflow/lite/testdata/multi_add.bin";
  private static final ByteBuffer MODEL_BUFFER = TestUtils.getTestFileAsBuffer(MODEL_PATH);

  @Test
  public void testBasic() throws Exception {
    try (GpuDelegate delegate = new GpuDelegate()) {
      assertThat(delegate.getNativeHandle()).isNotEqualTo(0);
    }
  }

  @Test
  public void testInterpreterWithGpu() throws Exception {
    Interpreter.Options options = new Interpreter.Options();
    try (GpuDelegate delegate = new GpuDelegate();
        Interpreter interpreter = new Interpreter(MODEL_BUFFER, options.addDelegate(delegate))) {
      float[] input0 = {1.23f};
      float[] input1 = {2.43f};
      Object[] inputs = {input0, input1, input0, input1};
      float[] parsedOutput0 = new float[1];
      float[] parsedOutput1 = new float[1];
      Map<Integer, Object> outputs = new HashMap<>();
      outputs.put(0, parsedOutput0);
      outputs.put(1, parsedOutput1);
      interpreter.runForMultipleInputsOutputs(inputs, outputs);
      float[] expected0 = {4.89f};
      float[] expected1 = {6.09f};
      assertThat(parsedOutput0).usingTolerance(0.1f).containsExactly(expected0).inOrder();
      assertThat(parsedOutput1).usingTolerance(0.1f).containsExactly(expected1).inOrder();
    }
  }
}
