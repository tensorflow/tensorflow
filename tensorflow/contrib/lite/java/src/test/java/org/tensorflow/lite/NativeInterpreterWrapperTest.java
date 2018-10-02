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
import static org.junit.Assert.fail;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.lite.NativeInterpreterWrapper}. */
// TODO(b/71818425): Generates model files dynamically.
@RunWith(JUnit4.class)
public final class NativeInterpreterWrapperTest {

  private static final String FLOAT_MODEL_PATH =
      "tensorflow/contrib/lite/java/src/testdata/add.bin";

  private static final String INT_MODEL_PATH =
      "tensorflow/contrib/lite/java/src/testdata/int32.bin";

  private static final String LONG_MODEL_PATH =
      "tensorflow/contrib/lite/java/src/testdata/int64.bin";

  private static final String BYTE_MODEL_PATH =
      "tensorflow/contrib/lite/java/src/testdata/uint8.bin";

  private static final String QUANTIZED_MODEL_PATH =
      "tensorflow/contrib/lite/java/src/testdata/quantized.bin";

  private static final String INVALID_MODEL_PATH =
      "tensorflow/contrib/lite/java/src/testdata/invalid_model.bin";

  private static final String MODEL_WITH_CUSTOM_OP_PATH =
      "tensorflow/contrib/lite/java/src/testdata/with_custom_op.lite";

  private static final String NONEXISTING_MODEL_PATH =
      "tensorflow/contrib/lite/java/src/testdata/nonexisting_model.bin";

  @Test
  public void testConstructor() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH);
    assertThat(wrapper).isNotNull();
    wrapper.close();
  }

  @Test
  public void testConstructorWithOptions() {
    NativeInterpreterWrapper wrapper =
        new NativeInterpreterWrapper(
            FLOAT_MODEL_PATH, new Interpreter.Options().setNumThreads(2).setUseNNAPI(true));
    assertThat(wrapper).isNotNull();
    wrapper.close();
  }

  @Test
  public void testConstructorWithInvalidModel() {
    try {
      NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(INVALID_MODEL_PATH);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("The model is not a valid Flatbuffer file");
    }
  }

  @Test
  public void testConstructorWithNonexistingModel() {
    try {
      NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(NONEXISTING_MODEL_PATH);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("The model is not a valid Flatbuffer file");
      assertThat(e).hasMessageThat().contains("Could not open");
    }
  }

  @Test
  public void testConstructorWithUnresolableCustomOp() {
    try {
      NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(MODEL_WITH_CUSTOM_OP_PATH);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("Cannot create interpreter: Didn't find custom op for name 'Assign'");
    }
  }

  @Test
  public void testRunWithFloat() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH);
    float[] oneD = {1.23f, -6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    float[][][][] fourD = {threeD, threeD};
    Object[] inputs = {fourD};
    float[][][][] parsedOutputs = new float[2][8][8][3];
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, parsedOutputs);
    wrapper.run(inputs, outputs);
    float[] outputOneD = parsedOutputs[0][0][0];
    float[] expected = {3.69f, -19.62f, 23.43f};
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
    wrapper.close();
  }

  @Test
  public void testRunWithBufferOutput() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH)) {
      float[] oneD = {1.23f, -6.54f, 7.81f};
      float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
      float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
      float[][][][] fourD = {threeD, threeD};
      Object[] inputs = {fourD};
      ByteBuffer parsedOutput =
          ByteBuffer.allocateDirect(2 * 8 * 8 * 3 * 4).order(ByteOrder.nativeOrder());
      Map<Integer, Object> outputs = new HashMap<>();
      outputs.put(0, parsedOutput);
      wrapper.run(inputs, outputs);
      float[] outputOneD = {
        parsedOutput.getFloat(0), parsedOutput.getFloat(4), parsedOutput.getFloat(8)
      };
      float[] expected = {3.69f, -19.62f, 23.43f};
      assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
    }
  }

  @Test
  public void testRunWithInputsOfSameDims() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH);
    float[] oneD = {1.23f, -6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    float[][][][] fourD = {threeD, threeD};
    Object[] inputs = {fourD};
    float[][][][] parsedOutputs = new float[2][8][8][3];
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, parsedOutputs);
    wrapper.run(inputs, outputs);
    float[] outputOneD = parsedOutputs[0][0][0];
    float[] expected = {3.69f, -19.62f, 23.43f};
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
    parsedOutputs = new float[2][8][8][3];
    outputs.put(0, parsedOutputs);
    wrapper.run(inputs, outputs);
    outputOneD = parsedOutputs[0][0][0];
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
    wrapper.close();
  }

  @Test
  public void testRunWithInt() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(INT_MODEL_PATH);
    int[] oneD = {3, 7, -4};
    int[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    int[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    int[][][][] fourD = {threeD, threeD};
    Object[] inputs = {fourD};
    int[][][][] parsedOutputs = new int[2][4][4][12];
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, parsedOutputs);
    wrapper.run(inputs, outputs);
    int[] outputOneD = parsedOutputs[0][0][0];
    int[] expected = {3, 7, -4, 3, 7, -4, 3, 7, -4, 3, 7, -4};
    assertThat(outputOneD).isEqualTo(expected);
    wrapper.close();
  }

  @Test
  public void testRunWithLong() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(LONG_MODEL_PATH);
    long[] oneD = {-892834092L, 923423L, 2123918239018L};
    long[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    long[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    long[][][][] fourD = {threeD, threeD};
    Object[] inputs = {fourD};
    long[][][][] parsedOutputs = new long[2][4][4][12];
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, parsedOutputs);
    wrapper.run(inputs, outputs);
    long[] outputOneD = parsedOutputs[0][0][0];
    long[] expected = {-892834092L, 923423L, 2123918239018L, -892834092L, 923423L, 2123918239018L,
                       -892834092L, 923423L, 2123918239018L, -892834092L, 923423L, 2123918239018L};
    assertThat(outputOneD).isEqualTo(expected);
    wrapper.close();
  }

  @Test
  public void testRunWithByte() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(BYTE_MODEL_PATH);
    byte[] oneD = {(byte) 0xe0, 0x4f, (byte) 0xd0};
    byte[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    byte[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    byte[][][][] fourD = {threeD, threeD};
    Object[] inputs = {fourD};
    int[] inputDims = {2, 8, 8, 3};
    wrapper.resizeInput(0, inputDims);
    byte[][][][] parsedOutputs = new byte[2][4][4][12];
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, parsedOutputs);
    wrapper.run(inputs, outputs);
    byte[] outputOneD = parsedOutputs[0][0][0];
    byte[] expected = {(byte) 0xe0, 0x4f, (byte) 0xd0, (byte) 0xe0, 0x4f, (byte) 0xd0,
                       (byte) 0xe0, 0x4f, (byte) 0xd0, (byte) 0xe0, 0x4f, (byte) 0xd0};
    assertThat(outputOneD).isEqualTo(expected);
    wrapper.close();
  }

  @Test
  public void testRunWithByteBufferHavingBytes() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(BYTE_MODEL_PATH);
    ByteBuffer bbuf = ByteBuffer.allocateDirect(2 * 8 * 8 * 3);
    bbuf.order(ByteOrder.nativeOrder());
    bbuf.rewind();
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 8; ++j) {
        for (int k = 0; k < 8; ++k) {
          bbuf.put((byte) 0xe0);
          bbuf.put((byte) 0x4f);
          bbuf.put((byte) 0xd0);
        }
      }
    }
    bbuf.rewind();
    Object[] inputs = {bbuf};
    int[] inputDims = {2, 8, 8, 3};
    wrapper.resizeInput(0, inputDims);
    byte[][][][] parsedOutputs = new byte[2][4][4][12];
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, parsedOutputs);
    wrapper.run(inputs, outputs);
    byte[] outputOneD = parsedOutputs[0][0][0];
    byte[] expected = {
      (byte) 0xe0, 0x4f, (byte) 0xd0, (byte) 0xe0, 0x4f, (byte) 0xd0,
      (byte) 0xe0, 0x4f, (byte) 0xd0, (byte) 0xe0, 0x4f, (byte) 0xd0
    };
    assertThat(outputOneD).isEqualTo(expected);
    wrapper.close();
  }

  @Test
  public void testRunWithByteBufferHavingFloats() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH);
    ByteBuffer bbuf = ByteBuffer.allocateDirect(4 * 8 * 8 * 3 * 4);
    bbuf.order(ByteOrder.nativeOrder());
    bbuf.rewind();
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 8; ++j) {
        for (int k = 0; k < 8; ++k) {
          bbuf.putFloat(1.23f);
          bbuf.putFloat(-6.54f);
          bbuf.putFloat(7.81f);
        }
      }
    }
    Object[] inputs = {bbuf};
    float[][][][] parsedOutputs = new float[4][8][8][3];
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, parsedOutputs);
    try {
      wrapper.run(inputs, outputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Cannot convert between a TensorFlowLite buffer with 768 bytes and a "
                  + "ByteBuffer with 3072 bytes.");
    }
    int[] inputDims = {4, 8, 8, 3};
    wrapper.resizeInput(0, inputDims);
    wrapper.run(inputs, outputs);
    float[] outputOneD = parsedOutputs[0][0][0];
    float[] expected = {3.69f, -19.62f, 23.43f};
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
    wrapper.close();
  }

  @Test
  public void testRunWithByteBufferHavingWrongSize() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(BYTE_MODEL_PATH);
    ByteBuffer bbuf = ByteBuffer.allocateDirect(2 * 7 * 8 * 3);
    bbuf.order(ByteOrder.nativeOrder());
    Object[] inputs = {bbuf};
    Map<Integer, Object> outputs = new HashMap<>();
    ByteBuffer parsedOutput = ByteBuffer.allocateDirect(2 * 7 * 8 * 3);
    outputs.put(0, parsedOutput);
    try {
      wrapper.run(inputs, outputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Cannot convert between a TensorFlowLite buffer with 192 bytes and a "
                  + "ByteBuffer with 336 bytes.");
    }
    wrapper.close();
  }

  @Test
  public void testRunWithWrongInputType() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH);
    int[] oneD = {4, 3, 9};
    int[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    int[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    int[][][][] fourD = {threeD, threeD};
    Object[] inputs = {fourD};
    int[][][][] parsedOutputs = new int[2][8][8][3];
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, parsedOutputs);
    try {
      wrapper.run(inputs, outputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Cannot convert between a TensorFlowLite tensor with type FLOAT32 and a Java object "
                  + "of type [[[[I (which is compatible with the TensorFlowLite type INT32)");
    }
    wrapper.close();
  }

  @Test
  public void testRunAfterClose() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH);
    wrapper.close();
    float[] oneD = {1.23f, 6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    float[][][][] fourD = {threeD, threeD};
    Object[] inputs = {fourD};
    float[][][][] parsedOutputs = new float[2][8][8][3];
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, parsedOutputs);
    try {
      wrapper.run(inputs, outputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("Invalid handle to Interpreter.");
    }
  }

  @Test
  public void testRunWithEmptyInputs() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH);
    try {
      Object[] inputs = {};
      wrapper.run(inputs, null);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("Inputs should not be null or empty.");
    }
    wrapper.close();
  }

  @Test
  public void testRunWithWrongInputSize() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH);
    float[] oneD = {1.23f, 6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    float[][][][] fourD = {threeD, threeD};
    Object[] inputs = {fourD, fourD};
    float[][][][] parsedOutputs = new float[2][8][8][3];
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, parsedOutputs);
    try {
      wrapper.run(inputs, outputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("Invalid input Tensor index: 1");
    }
    wrapper.close();
  }

  @Test
  public void testRunWithWrongInputNumOfDims() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH);
    float[] oneD = {1.23f, 6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    Object[] inputs = {threeD};
    float[][][][] parsedOutputs = new float[2][8][8][3];
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, parsedOutputs);
    try {
      wrapper.run(inputs, outputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Cannot copy between a TensorFlowLite tensor with shape [8, 7, 3] and a "
                  + "Java object with shape [2, 8, 8, 3].");
    }
    wrapper.close();
  }

  @Test
  public void testRunWithWrongInputDims() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH);
    float[] oneD = {1.23f, 6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    float[][][][] fourD = {threeD, threeD};
    Object[] inputs = {fourD};
    float[][][][] parsedOutputs = new float[2][8][8][3];
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, parsedOutputs);
    try {
      wrapper.run(inputs, outputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Cannot copy between a TensorFlowLite tensor with shape [2, 8, 7, 3] and a "
                  + "Java object with shape [2, 8, 8, 3].");
    }
    wrapper.close();
  }

  @Test
  public void testGetInferenceLatency() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH);
    float[] oneD = {1.23f, 6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    float[][][][] fourD = {threeD, threeD};
    Object[] inputs = {fourD};
    float[][][][] parsedOutputs = new float[2][8][8][3];
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, parsedOutputs);
    wrapper.run(inputs, outputs);
    assertThat(wrapper.getLastNativeInferenceDurationNanoseconds()).isGreaterThan(0L);
    wrapper.close();
  }

  @Test
  public void testGetInferenceLatencyWithNewWrapper() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH);
    assertThat(wrapper.getLastNativeInferenceDurationNanoseconds()).isNull();
    wrapper.close();
  }

  @Test
  public void testGetLatencyAfterFailedInference() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH);
    float[] oneD = {1.23f, 6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    float[][][][] fourD = {threeD, threeD};
    Object[] inputs = {fourD};
    float[][][][] parsedOutputs = new float[2][8][8][3];
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, parsedOutputs);
    try {
      wrapper.run(inputs, outputs);
      fail();
    } catch (IllegalArgumentException e) {
      // Expected.
    }
    assertThat(wrapper.getLastNativeInferenceDurationNanoseconds()).isNull();
    wrapper.close();
  }

  @Test
  public void testGetInputDims() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH);
    int[] expectedDims = {1, 8, 8, 3};
    assertThat(wrapper.getInputTensor(0).shape()).isEqualTo(expectedDims);
    wrapper.close();
  }

  @Test
  public void testGetOutputQuantizationParams() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH)) {
      assertThat(wrapper.getOutputQuantizationZeroPoint(0)).isEqualTo(0);
      assertThat(wrapper.getOutputQuantizationScale(0)).isWithin(1e-6f).of(0.0f);
    }
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(QUANTIZED_MODEL_PATH)) {
      assertThat(wrapper.getOutputQuantizationZeroPoint(0)).isEqualTo(127);
      assertThat(wrapper.getOutputQuantizationScale(0)).isWithin(1e-6f).of(0.25f);
    }
  }
}
