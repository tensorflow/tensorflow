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
import static org.junit.Assert.fail;

import java.nio.ByteBuffer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.lite.Interpreter} with selective registration. */
@RunWith(JUnit4.class)
public final class InterpreterCustomizedAndroidBuildTest {
  // Supported model.
  private static final String SUPPORTED_MODEL_PATH = "tensorflow/lite/testdata/add.bin";
  private static final ByteBuffer SUPPORTED_MODEL_BUFFER =
      TestUtils.getTestFileAsBuffer(SUPPORTED_MODEL_PATH);

  // Model with unregistered operator.
  private static final String UNSUPPORTED_MODEL_PATH =
      "tensorflow/lite/testdata/test_model.bin";
  private static final ByteBuffer UNSUPPORTED_MODEL_BUFFER =
      TestUtils.getTestFileAsBuffer(UNSUPPORTED_MODEL_PATH);

  @Test
  public void testSupportedModel() throws Exception {
    try (Interpreter interpreter = new Interpreter(SUPPORTED_MODEL_BUFFER)) {
      assertThat(interpreter).isNotNull();
      float[] oneD = {1.23f, 6.54f, 7.81f};
      float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
      float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
      float[][][][] fourD = {threeD, threeD};
      float[][][][] parsedOutputs = new float[2][8][8][3];
      interpreter.run(fourD, parsedOutputs);
    }
  }

  @Test
  public void testUnsupportedModel() throws Exception {
    try (Interpreter interpreter = new Interpreter(UNSUPPORTED_MODEL_BUFFER)) {
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("Cannot create interpreter: Didn't find op for builtin opcode 'CONV_2D'");
    }
  }
}
