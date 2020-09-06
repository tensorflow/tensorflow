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

import java.nio.ByteBuffer;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Map;
import java.util.PriorityQueue;
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

  private static final ByteBuffer MOBILENET_FLOAT_MODEL_BUFFER =
      TestUtils.getTestFileAsBuffer(
          "third_party/tensorflow/lite/java/demo/app/src/main/assets/mobilenet_v1_1.0_224.tflite");

  private static final ByteBuffer MOBILENET_QUANTIZED_MODEL_BUFFER =
      TestUtils.getTestFileAsBuffer(
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
  public void testMobileNetEnhancedCpuKernels() {
    runMobileNetFloatTest(new Interpreter.Options().setUseXNNPACK(true));
  }

  @Test
  public void testMobileNetEnhancedCpuKernelsMultithreaded() {
    runMobileNetFloatTest(new Interpreter.Options().setUseXNNPACK(true).setNumThreads(2));
  }

  @Test
  public void testMobileNetQuantized() {
    runMobileNetQuantizedTest(new Interpreter.Options());
  }

  @Test
  public void testMobileNetQuantizedMultithreaded() {
    runMobileNetQuantizedTest(new Interpreter.Options().setNumThreads(2));
  }

  @Test
  public void testMobileNetQuantizedEnhancedCpu() {
    // The "enhanced CPU flag" should only impact float models, this is a sanity test to confirm.
    runMobileNetQuantizedTest(new Interpreter.Options().setUseXNNPACK(true));
  }

  private static void runMobileNetFloatTest(Interpreter.Options options) {
    ByteBuffer img =
        TestUtils.getTestImageAsFloatByteBuffer(
            "tensorflow/lite/java/src/testdata/grace_hopper_224.jpg");
    float[][] labels = new float[1][1001];
    try (Interpreter interpreter = new Interpreter(MOBILENET_FLOAT_MODEL_BUFFER, options)) {
      interpreter.run(img, labels);
      assertThat(interpreter.getInputTensor(0).shape()).isEqualTo(new int[] {1, 224, 224, 3});
      assertThat(interpreter.getOutputTensor(0).shape()).isEqualTo(new int[] {1, 1001});
    }
    assertThat(labels[0])
        .usingExactEquality()
        .containsNoneOf(new float[] {Float.NaN, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY});
    // 653 == "military uniform"
    assertThat(getTopKLabels(labels, 3)).contains(653);
  }

  private static void runMobileNetQuantizedTest(Interpreter.Options options) {
    ByteBuffer img =
        TestUtils.getTestImageAsByteBuffer(
            "tensorflow/lite/java/src/testdata/grace_hopper_224.jpg");
    byte[][] labels = new byte[1][1001];
    try (Interpreter interpreter = new Interpreter(MOBILENET_QUANTIZED_MODEL_BUFFER, options)) {
      interpreter.run(img, labels);
      assertThat(interpreter.getInputTensor(0).shape()).isEqualTo(new int[] {1, 224, 224, 3});
      assertThat(interpreter.getOutputTensor(0).shape()).isEqualTo(new int[] {1, 1001});
    }
    // 653 == "military uniform"
    assertThat(getTopKLabels(labels, 3)).contains(653);
  }

  private static ArrayList<Integer> getTopKLabels(byte[][] byteLabels, int k) {
    float[][] labels = new float[1][1001];
    for (int i = 0; i < byteLabels[0].length; ++i) {
      labels[0][i] = (byteLabels[0][i] & 0xff) / 255.0f;
    }
    return getTopKLabels(labels, k);
  }

  private static ArrayList<Integer> getTopKLabels(float[][] labels, int k) {
    PriorityQueue<Map.Entry<Integer, Float>> pq =
        new PriorityQueue<>(
            k,
            new Comparator<Map.Entry<Integer, Float>>() {
              @Override
              public int compare(Map.Entry<Integer, Float> o1, Map.Entry<Integer, Float> o2) {
                // Intentionally reversed to put high confidence at the head of the queue.
                return o1.getValue().compareTo(o2.getValue()) * -1;
              }
            });

    for (int i = 0; i < labels[0].length; ++i) {
      pq.add(new AbstractMap.SimpleEntry<>(i, labels[0][i]));
    }

    final ArrayList<Integer> topKLabels = new ArrayList<>();
    int topKLabelsSize = Math.min(pq.size(), k);
    for (int i = 0; i < topKLabelsSize; ++i) {
      topKLabels.add(pq.poll().getKey());
    }
    return topKLabels;
  }
}
