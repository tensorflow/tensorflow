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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Smoke tests for {@link org.tensorflow.lite.Interpreter} agains MobileNet models.
 *
 * <p>Note that these tests are not intended to validate accuracy, rather, they serve to exercise
 * end-to-end inference, against a meaningful model, to tease out any stability/runtime issues.
 */
@RunWith(JUnit4.class)
public final class InterpreterMobileNetTest {

  private static final File MOBILENET_FLOAT_MODEL_FILE =
      new File("tensorflow/lite/java/src/testdata/mobilenet.tflite.bin");

  private static final File MOBILENET_QUANTIZED_MODEL_FILE =
      new File(
          "third_party/tensorflow/lite/java/demo/app/src/main/assets/mobilenet_v1_1.0_224_quant.tflite");

  @Test
  public void testMobileNet() {
    runMobileNetFloatTest(new Interpreter.Options());
  }

  @Test
  public void testMobileNetMultithreaded() {
    runMobileNetFloatTest(new Interpreter.Options().setNumThreads(2));
  }

  @Test
  public void testMobileNetQuantized() {
    runMobileNetQuantizedTest(new Interpreter.Options());
  }

  @Test
  public void testMobileNetQuantizedMultithreaded() {
    runMobileNetQuantizedTest(new Interpreter.Options().setNumThreads(2));
  }

  private static void runMobileNetFloatTest(Interpreter.Options options) {
    // Create a gray image.
    ByteBuffer img = ByteBuffer.allocateDirect(1 * 224 * 224 * 3 * 4);
    img.order(ByteOrder.nativeOrder());
    img.rewind();
    while (img.hasRemaining()) {
      img.putFloat(0.5f);
    }

    float[][] labels = new float[1][1001];
    try (Interpreter interpreter = new Interpreter(MOBILENET_FLOAT_MODEL_FILE, options)) {
      interpreter.run(img, labels);
      assertThat(interpreter.getInputTensor(0).shape()).isEqualTo(new int[] {1, 224, 224, 3});
      assertThat(interpreter.getOutputTensor(0).shape()).isEqualTo(new int[] {1, 1001});
    }
    assertThat(labels[0])
        .usingExactEquality()
        .containsNoneOf(new float[] {Float.NaN, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY});
  }

  private static void runMobileNetQuantizedTest(Interpreter.Options options) {
    // Create a gray image.
    ByteBuffer img = ByteBuffer.allocateDirect(1 * 224 * 224 * 3);
    img.order(ByteOrder.nativeOrder());
    img.rewind();
    while (img.hasRemaining()) {
      img.put((byte) 128);
    }

    try (Interpreter interpreter = new Interpreter(MOBILENET_QUANTIZED_MODEL_FILE, options)) {
      byte[][] labels = new byte[1][1001];
      interpreter.run(img, labels);
      assertThat(interpreter.getInputTensor(0).shape()).isEqualTo(new int[] {1, 224, 224, 3});
      assertThat(interpreter.getOutputTensor(0).shape()).isEqualTo(new int[] {1, 1001});
    }
  }
}
