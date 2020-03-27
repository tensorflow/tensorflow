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
      "tensorflow/lite/java/src/testdata/add.bin";

  private static final String INT_MODEL_PATH =
      "tensorflow/lite/java/src/testdata/int32.bin";

  private static final String LONG_MODEL_PATH =
      "tensorflow/lite/java/src/testdata/int64.bin";

  private static final String BYTE_MODEL_PATH =
      "tensorflow/lite/java/src/testdata/uint8.bin";

  private static final String STRING_MODEL_PATH =
      "tensorflow/lite/java/src/testdata/string.bin";

  private static final String INVALID_MODEL_PATH =
      "tensorflow/lite/java/src/testdata/invalid_model.bin";

  private static final String MODEL_WITH_CUSTOM_OP_PATH =
      "tensorflow/lite/java/src/testdata/with_custom_op.lite";

  private static final String NONEXISTING_MODEL_PATH =
      "tensorflow/lite/java/src/testdata/nonexisting_model.bin";

  @Test
  public void testConstructor() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH)) {
      assertThat(wrapper).isNotNull();
    }
  }

  @Test
  public void testConstructorWithOptions() {
    try (NativeInterpreterWrapper wrapper =
        new NativeInterpreterWrapper(
            FLOAT_MODEL_PATH, new Interpreter.Options().setNumThreads(2).setUseNNAPI(true))) {
      assertThat(wrapper).isNotNull();
    }
  }

  @Test
  public void testConstructorWithInvalidModel() {
    try {
      @SuppressWarnings("unused")
      NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(INVALID_MODEL_PATH);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("The model is not a valid Flatbuffer file");
    }
  }

  @Test
  public void testConstructorWithNonexistingModel() {
    try {
      @SuppressWarnings("unused")
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
      @SuppressWarnings("unused")
      NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(MODEL_WITH_CUSTOM_OP_PATH);
      fail();
    } catch (IllegalStateException e) {
      assertThat(e).hasMessageThat().contains("Encountered unresolved custom op: Assign");
    }
  }

  @Test
  public void testRunWithFloat() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH)) {
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
    }
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
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH)) {
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
    }
  }

  @Test
  public void testRunWithInt() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(INT_MODEL_PATH)) {
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
    }
  }

  @Test
  public void testRunWithLong() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(LONG_MODEL_PATH)) {
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
    }
  }

  @Test
  public void testRunWithByte() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(BYTE_MODEL_PATH)) {
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
    }
  }

  @Test
  public void testRunWithString() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(STRING_MODEL_PATH)) {
      String[] oneD = {"s1", "s22", "s333"};
      String[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
      String[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
      String[][][][] fourD = {threeD, threeD};
      Object[] inputs = {fourD};
      String[][][][] parsedOutputs = new String[2][4][4][12];
      Map<Integer, Object> outputs = new HashMap<>();
      outputs.put(0, parsedOutputs);
      wrapper.run(inputs, outputs);
      String[] outputOneD = parsedOutputs[0][0][0];
      String[] expected = {
          "s1", "s22", "s333", "s1", "s22", "s333", "s1", "s22", "s333", "s1", "s22", "s333"
      };
      assertThat(outputOneD).isEqualTo(expected);
    }
  }

  @Test
  public void testRunWithString_supplementaryUnicodeCharacters() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(STRING_MODEL_PATH)) {
      String[] oneD = {"\uD800\uDC01", "s22", "\ud841\udf0e"};
      String[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
      String[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
      String[][][][] fourD = {threeD, threeD};
      Object[] inputs = {fourD};
      String[][][][] parsedOutputs = new String[2][4][4][12];
      Map<Integer, Object> outputs = new HashMap<>();
      outputs.put(0, parsedOutputs);
      wrapper.run(inputs, outputs);
      String[] outputOneD = parsedOutputs[0][0][0];
      String[] expected = {
          "\uD800\uDC01", "s22", "\ud841\udf0e", "\uD800\uDC01", "s22", "\ud841\udf0e",
          "\uD800\uDC01", "s22", "\ud841\udf0e", "\uD800\uDC01", "s22", "\ud841\udf0e"
      };
      assertThat(outputOneD).isEqualTo(expected);
    }
  }

  @Test
  public void testRunWithString_wrongShapeError() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(STRING_MODEL_PATH)) {
      String[] oneD = {"s1", "s22", "s333"};
      String[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
      String[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
      String[][][][] fourD = {threeD, threeD};
      Object[] inputs = {fourD};
      String[][][][] parsedOutputs = new String[2][4][4][10];
      Map<Integer, Object> outputs = new HashMap<>();
      outputs.put(0, parsedOutputs);
      try {
        wrapper.run(inputs, outputs);
        fail();
      } catch (IllegalArgumentException e) {
        assertThat(e)
            .hasMessageThat()
            .contains(
                "Cannot copy between a TensorFlowLite tensor with shape [2, 4, 4, 12] and "
                    + "a Java object with shape [2, 4, 4, 10]");
      }
    }
  }

  @Test
  public void testRunWithByteBufferHavingBytes() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(BYTE_MODEL_PATH)) {
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
    }
  }

  @Test
  public void testRunWithByteBufferHavingFloats() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH)) {
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
                    + "Java Buffer with 3072 bytes.");
      }
      int[] inputDims = {4, 8, 8, 3};
      wrapper.resizeInput(0, inputDims);
      wrapper.run(inputs, outputs);
      float[] outputOneD = parsedOutputs[0][0][0];
      float[] expected = {3.69f, -19.62f, 23.43f};
      assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
    }
  }

  @Test
  public void testRunWithByteBufferHavingWrongSize() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(BYTE_MODEL_PATH)) {
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
                    + "Java Buffer with 336 bytes.");
      }
    }
  }

  @Test
  public void testRunWithWrongInputType() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH)) {
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
                "Cannot convert between a TensorFlowLite tensor with type FLOAT32 and a Java "
                    + "object of type [[[[I (which is compatible with the TensorFlowLite type "
                    + "INT32)");
      }
    }
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
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH)) {
      try {
        Object[] inputs = {};
        wrapper.run(inputs, null);
        fail();
      } catch (IllegalArgumentException e) {
        assertThat(e).hasMessageThat().contains("Inputs should not be null or empty.");
      }
    }
  }

  @Test
  public void testRunWithWrongInputSize() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH)) {
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
    }
  }

  @Test
  public void testRunWithWrongInputNumOfDims() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH)) {
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
    }
  }

  @Test
  public void testRunWithWrongInputDims() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH)) {
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
    }
  }

  @Test
  public void testGetInferenceLatency() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH)) {
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
    }
  }

  @Test
  public void testGetInferenceLatencyWithNewWrapper() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH)) {
      assertThat(wrapper.getLastNativeInferenceDurationNanoseconds()).isNull();
    }
  }

  @Test
  public void testGetLatencyAfterFailedInference() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH)) {
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
    }
  }

  @Test
  public void testGetInputDims() {
    try (NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH)) {
      int[] expectedDims = {1, 8, 8, 3};
      assertThat(wrapper.getInputTensor(0).shape()).isEqualTo(expectedDims);
    }
  }
}
