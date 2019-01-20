/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.lite.Interpreter} agains a MobileNet model. */
@RunWith(JUnit4.class)
public final class InterpreterMobileNetTest {

  private static final File MOBILENET_MODEL_FILE =
      new File("tensorflow/lite/java/src/testdata/mobilenet.tflite.bin");

  @Test
  public void testMobilenetRun() {
    // Create a gray image.
    float[][][][] img = new float[1][224][224][3];
    for (int i = 0; i < 224; ++i) {
      for (int j = 0; j < 224; ++j) {
        img[0][i][j][0] = 0.5f;
        img[0][i][j][1] = 0.5f;
        img[0][i][j][2] = 0.5f;
      }
    }

    // Allocate memory to receive the output values.
    float[][] labels = new float[1][1001];

    Interpreter interpreter = new Interpreter(MOBILENET_MODEL_FILE);
    interpreter.run(img, labels);
    assertThat(interpreter.getInputTensor(0).shape()).isEqualTo(new int[] {1, 224, 224, 3});
    assertThat(interpreter.getOutputTensor(0).shape()).isEqualTo(new int[] {1, 1001});
    interpreter.close();

    assertThat(labels[0])
        .usingExactEquality()
        .containsNoneOf(new float[] {Float.NaN, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY});
  }
}
