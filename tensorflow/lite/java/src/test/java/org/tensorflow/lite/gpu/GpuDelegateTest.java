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
import static java.util.concurrent.TimeUnit.MICROSECONDS;
import static org.tensorflow.lite.gpu.GpuDelegateFactory.Options.GpuBackend.OPENCL;

import com.google.common.base.Stopwatch;
import java.io.File;
import java.nio.ByteBuffer;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.InterpreterTestHelper;
import org.tensorflow.lite.TestUtils;

/** Unit tests for {@link org.tensorflow.lite.gpu.GpuDelegate}. */
@RunWith(JUnit4.class)
public final class GpuDelegateTest {

  private static final String MODEL_PATH = "tensorflow/lite/testdata/multi_add.bin";
  private static final ByteBuffer MODEL_BUFFER = TestUtils.getTestFileAsBuffer(MODEL_PATH);
  private static final ByteBuffer MOBILENET_QUANTIZED_MODEL_BUFFER =
      TestUtils.getTestFileAsBuffer(
          "tensorflow/lite/java/demo/app/src/main/assets/mobilenet_v1_1.0_224_quant.tflite");

  @Rule public final TemporaryFolder tempDir = new TemporaryFolder();

  @Test
  public void testBasic() throws Exception {
    try (GpuDelegate delegate = new GpuDelegate()) {
      assertThat(delegate.getNativeHandle()).isNotEqualTo(0);
    }
  }

  @Test
  public void testInterpreterWithGpu_FloatModel() throws Exception {
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

  @Test
  public void testInterpreterWithGpu_QuantModelRunWithDelegate() throws Exception {
    ByteBuffer img =
        TestUtils.getTestImageAsByteBuffer(
            "tensorflow/lite/java/src/testdata/grace_hopper_224.jpg");

    Interpreter.Options options = new Interpreter.Options();
    // Default behavior allows quantized models.
    try (GpuDelegate delegate = new GpuDelegate();
        Interpreter interpreter =
            new Interpreter(MOBILENET_QUANTIZED_MODEL_BUFFER, options.addDelegate(delegate))) {
      byte[][] output = new byte[1][1001];
      interpreter.run(img, output);
      // Should be only 1 node (Delegate) in the execution plan.
      assertThat(InterpreterTestHelper.executionPlanLength(interpreter)).isEqualTo(1);
      assertThat(interpreter.getInputTensor(0).shape()).isEqualTo(new int[] {1, 224, 224, 3});
      assertThat(interpreter.getOutputTensor(0).shape()).isEqualTo(new int[] {1, 1001});
      // 653 == "military uniform"
      assertThat(getTopKLabels(output, 3)).contains(653);
    }
  }

  @Test
  public void testInterpreterWithGpu_QuantModelRunOnCPU() throws Exception {
    ByteBuffer img =
        TestUtils.getTestImageAsByteBuffer(
            "tensorflow/lite/java/src/testdata/grace_hopper_224.jpg");

    Interpreter.Options options = new Interpreter.Options();
    try (GpuDelegate delegate =
            new GpuDelegate(new GpuDelegateFactory.Options().setQuantizedModelsAllowed(false));
        Interpreter interpreter =
            new Interpreter(MOBILENET_QUANTIZED_MODEL_BUFFER, options.addDelegate(delegate))) {
      byte[][] output = new byte[1][1001];
      interpreter.run(img, output);
      // Model is now delegated to XNNPACK as the GPU delegate was deactivated.
      // Don't hard code the execution plan length as this may change as more operators are added to
      // XNNPACK.
      assertThat(InterpreterTestHelper.executionPlanLength(interpreter)).isLessThan(31);
      assertThat(interpreter.getInputTensor(0).shape()).isEqualTo(new int[] {1, 224, 224, 3});
      assertThat(interpreter.getOutputTensor(0).shape()).isEqualTo(new int[] {1, 1001});
      // 653 == "military uniform"
      assertThat(getTopKLabels(output, 3)).contains(653);
    }
  }

  @Test
  public void testInterpreterWithGpu_forceOpenCl_worksOrThrowsException() {
    Interpreter.Options options = new Interpreter.Options();
    try (GpuDelegate delegate =
            new GpuDelegate(new GpuDelegateFactory.Options().setForceBackend(OPENCL));
         Interpreter interpreter =
            new Interpreter(MOBILENET_QUANTIZED_MODEL_BUFFER, options.addDelegate(delegate))) {
      ByteBuffer img =
        TestUtils.getTestImageAsByteBuffer(
            "tensorflow/lite/java/src/testdata/grace_hopper_224.jpg");
      byte[][] output = new byte[1][1001];
      interpreter.run(img, output);
      assertThat(interpreter.getInputTensor(0).shape()).isEqualTo(new int[] {1, 224, 224, 3});
      assertThat(interpreter.getOutputTensor(0).shape()).isEqualTo(new int[] {1, 1001});
      // 653 == "military uniform"
      assertThat(getTopKLabels(output, 3)).contains(653);
    } catch (IllegalArgumentException e) {
      // May fail if OpenCL is not available on the device.
      assertThat(e).hasMessageThat().contains("Can not open OpenCL library");
    }
  }

  @Test
  public void testDelegateSerialization() throws Exception {
    ByteBuffer img =
        TestUtils.getTestImageAsByteBuffer(
            "tensorflow/lite/java/src/testdata/grace_hopper_224.jpg");

    File serializationFolder = tempDir.newFolder();
    String serializationDir = serializationFolder.getPath();

    // Create the interpreter with serialization enabled delegate.
    createInterpreterWithDelegate(/*enableSerialization=*/ true, serializationFolder.getPath());

    // In the second interpreter initialization, delegate reuses the serialization data.
    Stopwatch stopWatch = Stopwatch.createStarted();
    Interpreter interpreter =
        createInterpreterWithDelegate(/*enableSerialization=*/ true, serializationFolder.getPath());
    stopWatch.stop();
    long serializedInitTime = stopWatch.elapsed(MICROSECONDS);
    // Check on the model.
    byte[][] output = new byte[1][1001];
    interpreter.run(img, output);
    // 653 == "military uniform"
    assertThat(getTopKLabels(output, 3)).contains(653);

    // If OpenCL is available, serialized data will be written to serializationDir and
    // initialization time improvement shall be observed.
    // Otherwise, this testcase performs a check that enabling the option won't crash.
    if (serializationFolder.list().length > 0) {
      stopWatch.reset();
      stopWatch.start();
      // Initialze interpreter with GpuDelegate serialization not enabled.
      createInterpreterWithDelegate(/*enableSerialization=*/ false, /*serializationDir=*/ null);
      long notserializedInitTime = stopWatch.elapsed(MICROSECONDS);

      assertThat(serializedInitTime).isLessThan(notserializedInitTime);
    }
  }

  private Interpreter createInterpreterWithDelegate(
      boolean enableSerialization, String serializationDir) {
    Interpreter.Options options = new Interpreter.Options();
    if (enableSerialization) {
      options.addDelegate(
          new GpuDelegate(
              new GpuDelegateFactory.Options()
                  .setSerializationParams(serializationDir, "GpuDelegateTest.testModelToken")));
    } else {
      options.addDelegate(new GpuDelegate());
    }
    Interpreter interpreter = new Interpreter(MOBILENET_QUANTIZED_MODEL_BUFFER, options);
    return interpreter;
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
