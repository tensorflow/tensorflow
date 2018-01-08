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
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.lite.NativeInterpreterWrapper}. */
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

  private static final String INVALID_MODEL_PATH =
      "tensorflow/contrib/lite/java/src/testdata/invalid_model.bin";

  @Test
  public void testConstructor() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(FLOAT_MODEL_PATH);
    assertThat(wrapper).isNotNull();
    wrapper.close();
  }

  @Test
  public void testConstructorWithInvalidModel() {
    try {
      NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(INVALID_MODEL_PATH);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("Model provided has model identifier ' is ', should be 'TFL3'");
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
    Tensor[] outputs = wrapper.run(inputs);
    assertThat(outputs.length).isEqualTo(1);
    float[][][][] parsedOutputs = new float[2][8][8][3];
    outputs[0].copyTo(parsedOutputs);
    float[] outputOneD = parsedOutputs[0][0][0];
    float[] expected = {3.69f, -19.62f, 23.43f};
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
    Tensor[] outputs = wrapper.run(inputs);
    assertThat(outputs.length).isEqualTo(1);
    int[][][][] parsedOutputs = new int[2][4][4][12];
    outputs[0].copyTo(parsedOutputs);
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
    Tensor[] outputs = wrapper.run(inputs);
    assertThat(outputs.length).isEqualTo(1);
    long[][][][] parsedOutputs = new long[2][4][4][12];
    outputs[0].copyTo(parsedOutputs);
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
    Tensor[] outputs = wrapper.run(inputs);
    assertThat(outputs.length).isEqualTo(1);
    byte[][][][] parsedOutputs = new byte[2][4][4][12];
    outputs[0].copyTo(parsedOutputs);
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
    Object[] inputs = {bbuf};
    int[] inputDims = {2, 8, 8, 3};
    wrapper.resizeInput(0, inputDims);
    Tensor[] outputs = wrapper.run(inputs);
    assertThat(outputs.length).isEqualTo(1);
    byte[][][][] parsedOutputs = new byte[2][4][4][12];
    outputs[0].copyTo(parsedOutputs);
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
    try {
      wrapper.run(inputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Failed to get input dimensions. 0-th input should have 768 bytes, but found 3072 bytes");
    }
    int[] inputDims = {4, 8, 8, 3};
    wrapper.resizeInput(0, inputDims);
    Tensor[] outputs = wrapper.run(inputs);
    assertThat(outputs.length).isEqualTo(1);
    float[][][][] parsedOutputs = new float[4][8][8][3];
    outputs[0].copyTo(parsedOutputs);
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
    try {
      wrapper.run(inputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Failed to get input dimensions. 0-th input should have 192 bytes, but found 336 bytes.");
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
    try {
      wrapper.run(inputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "DataType (2) of input data does not match with the DataType (1) of model inputs.");
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
    try {
      wrapper.run(inputs);
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
      wrapper.run(inputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("Invalid inputs. Inputs should not be null or empty.");
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
    try {
      wrapper.run(inputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("Expected num of inputs is 1 but got 2");
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
    try {
      wrapper.run(inputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("0-th input should have 4 dimensions, but found 3 dimensions");
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
    try {
      wrapper.run(inputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("0-th input dimension should be [?,8,8,3], but found [?,8,7,3]");
    }
    wrapper.close();
  }

  @Test
  public void testNumElements() {
    int[] shape = {2, 3, 4};
    int num = NativeInterpreterWrapper.numElements(shape);
    assertThat(num).isEqualTo(24);
    shape = null;
    num = NativeInterpreterWrapper.numElements(shape);
    assertThat(num).isEqualTo(0);
  }

  @Test
  public void testIsNonEmtpyArray() {
    assertThat(NativeInterpreterWrapper.isNonEmptyArray(null)).isFalse();
    assertThat(NativeInterpreterWrapper.isNonEmptyArray(3.2)).isFalse();
    int[] emptyArray = {};
    assertThat(NativeInterpreterWrapper.isNonEmptyArray(emptyArray)).isFalse();
    int[] validArray = {9, 5, 2, 1};
    assertThat(NativeInterpreterWrapper.isNonEmptyArray(validArray)).isTrue();
  }

  @Test
  public void testDataTypeOf() {
    float[] testEmtpyArray = {};
    DataType dataType = NativeInterpreterWrapper.dataTypeOf(testEmtpyArray);
    assertThat(dataType).isEqualTo(DataType.FLOAT32);
    float[] testFloatArray = {0.783f, 0.251f};
    dataType = NativeInterpreterWrapper.dataTypeOf(testFloatArray);
    assertThat(dataType).isEqualTo(DataType.FLOAT32);
    float[][] testMultiDimArray = {testFloatArray, testFloatArray, testFloatArray};
    dataType = NativeInterpreterWrapper.dataTypeOf(testFloatArray);
    assertThat(dataType).isEqualTo(DataType.FLOAT32);
    try {
      double[] testDoubleArray = {0.783, 0.251};
      NativeInterpreterWrapper.dataTypeOf(testDoubleArray);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("cannot resolve DataType of");
    }
    try {
      Float[] testBoxedArray = {0.783f, 0.251f};
      NativeInterpreterWrapper.dataTypeOf(testBoxedArray);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("cannot resolve DataType of [Ljava.lang.Float;");
    }
  }

  @Test
  public void testNumDimensions() {
    int scalar = 1;
    assertThat(NativeInterpreterWrapper.numDimensions(scalar)).isEqualTo(0);
    int[][] array = {{2, 4}, {1, 9}};
    assertThat(NativeInterpreterWrapper.numDimensions(array)).isEqualTo(2);
    try {
      int[] emptyArray = {};
      NativeInterpreterWrapper.numDimensions(emptyArray);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("array lengths cannot be 0.");
    }
  }

  @Test
  public void testFillShape() {
    int[][][] array = {{{23}, {14}, {87}}, {{12}, {42}, {31}}};
    int num = NativeInterpreterWrapper.numDimensions(array);
    int[] shape = new int[num];
    NativeInterpreterWrapper.fillShape(array, 0, shape);
    assertThat(num).isEqualTo(3);
    assertThat(shape[0]).isEqualTo(2);
    assertThat(shape[1]).isEqualTo(3);
    assertThat(shape[2]).isEqualTo(1);
  }
}
