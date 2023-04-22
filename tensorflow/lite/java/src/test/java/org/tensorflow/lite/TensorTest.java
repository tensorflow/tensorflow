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

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ReadOnlyBufferException;
import java.util.HashMap;
import java.util.Map;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.lite.Tensor.QuantizationParams;

/** Unit tests for {@link TensorImpl}. */
@RunWith(JUnit4.class)
public final class TensorTest {

  private static final String MODEL_PATH =
      "tensorflow/lite/java/src/testdata/add.bin";

  private static final String INT_MODEL_PATH =
      "tensorflow/lite/java/src/testdata/int32.bin";

  private static final String LONG_MODEL_PATH =
      "tensorflow/lite/java/src/testdata/int64.bin";

  private static final String STRING_MODEL_PATH =
      "tensorflow/lite/java/src/testdata/string.bin";

  private static final String QUANTIZED_MODEL_PATH =
      "tensorflow/lite/java/src/testdata/quantized.bin";

  private NativeInterpreterWrapper wrapper;
  private TensorImpl tensor;

  @Before
  public void setUp() {
    TestInit.init();
    wrapper = new NativeInterpreterWrapper(MODEL_PATH);
    float[] oneD = {1.23f, 6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    float[][][][] fourD = {threeD, threeD};
    Object[] inputs = {fourD};
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, new float[2][8][8][3]);
    wrapper.run(inputs, outputs);
    tensor = wrapper.getOutputTensor(0);
    assertThat(tensor.index()).isGreaterThan(-1);
  }

  @After
  public void tearDown() {
    wrapper.close();
  }

  @Test
  public void testBasic() throws Exception {
    assertThat(tensor).isNotNull();
    int[] expectedShape = {2, 8, 8, 3};
    assertThat(tensor.shape()).isEqualTo(expectedShape);
    assertThat(tensor.shapeSignature()).isEqualTo(expectedShape);
    assertThat(tensor.dataType()).isEqualTo(DataType.FLOAT32);
    assertThat(tensor.numBytes()).isEqualTo(2 * 8 * 8 * 3 * 4);
    assertThat(tensor.numElements()).isEqualTo(2 * 8 * 8 * 3);
    assertThat(tensor.numDimensions()).isEqualTo(4);
    assertThat(tensor.name()).isEqualTo("output");
    assertThat(tensor.asReadOnlyBuffer().capacity()).isEqualTo(tensor.numBytes());
  }

  @Test
  public void testCopyTo() {
    float[][][][] parsedOutputs = new float[2][8][8][3];
    tensor.copyTo(parsedOutputs);
    float[] outputOneD = parsedOutputs[0][0][0];
    float[] expected = {3.69f, 19.62f, 23.43f};
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
  }

  @Test
  public void testCopyToNull() {
    try {
      tensor.copyTo(null);
      fail();
    } catch (IllegalArgumentException e) {
      // Success.
    }
  }

  @Test
  public void testModifyReadOnlyBuffer() {
    try {
      assertThat(tensor.asReadOnlyBuffer().putFloat(0.f));
      fail();
    } catch (ReadOnlyBufferException e) {
      // Success.
    }
  }

  @Test
  public void testCopyToByteBuffer() {
    ByteBuffer parsedOutput =
        ByteBuffer.allocateDirect(2 * 8 * 8 * 3 * 4).order(ByteOrder.nativeOrder());
    tensor.copyTo(parsedOutput);
    assertThat(parsedOutput.position()).isEqualTo(2 * 8 * 8 * 3 * 4);
    float[] outputOneD = {
      parsedOutput.getFloat(0), parsedOutput.getFloat(4), parsedOutput.getFloat(8)
    };
    float[] expected = {3.69f, 19.62f, 23.43f};
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
  }

  @Test
  public void testCopyToLargerByteBuffer() {
    // Allocate a ByteBuffer that is larger than the Tensor, and ensure we can copy to it.
    ByteBuffer parsedOutput =
        ByteBuffer.allocateDirect(10 * 2 * 8 * 8 * 3 * 4).order(ByteOrder.nativeOrder());
    tensor.copyTo(parsedOutput);
    assertThat(parsedOutput.position()).isEqualTo(2 * 8 * 8 * 3 * 4);
    float[] outputOneD = {
      parsedOutput.getFloat(0), parsedOutput.getFloat(4), parsedOutput.getFloat(8)
    };
    float[] expected = {3.69f, 19.62f, 23.43f};
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
  }

  @Test
  public void testCopyToByteBufferAsFloatBuffer() {
    FloatBuffer parsedOutput =
        ByteBuffer.allocateDirect(2 * 8 * 8 * 3 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
    tensor.copyTo(parsedOutput);
    assertThat(parsedOutput.position()).isEqualTo(2 * 8 * 8 * 3);
    float[] outputOneD = {parsedOutput.get(0), parsedOutput.get(1), parsedOutput.get(2)};
    float[] expected = {3.69f, 19.62f, 23.43f};
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
  }

  @Test
  public void testCopyToFloatBuffer() {
    FloatBuffer parsedOutput = FloatBuffer.allocate(2 * 8 * 8 * 3);
    tensor.copyTo(parsedOutput);
    assertThat(parsedOutput.position()).isEqualTo(2 * 8 * 8 * 3);
    float[] outputOneD = {parsedOutput.get(0), parsedOutput.get(1), parsedOutput.get(2)};
    float[] expected = {3.69f, 19.62f, 23.43f};
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
  }

  @Test
  public void testCopyToIntBuffer() {
    wrapper = new NativeInterpreterWrapper(INT_MODEL_PATH);
    tensor = wrapper.getOutputTensor(0);
    IntBuffer parsedOutput = IntBuffer.allocate(1 * 4 * 4 * 12);
    tensor.copyTo(parsedOutput);
    assertThat(parsedOutput.position()).isEqualTo(1 * 4 * 4 * 12);
  }

  @Test
  public void testCopyToLongBuffer() {
    wrapper = new NativeInterpreterWrapper(LONG_MODEL_PATH);
    tensor = wrapper.getOutputTensor(0);
    LongBuffer parsedOutput = LongBuffer.allocate(1 * 4 * 4 * 12);
    tensor.copyTo(parsedOutput);
    assertThat(parsedOutput.position()).isEqualTo(1 * 4 * 4 * 12);
  }

  @Test
  public void testCopyToInvalidByteBuffer() {
    ByteBuffer parsedOutput = ByteBuffer.allocateDirect(3 * 4).order(ByteOrder.nativeOrder());
    try {
      tensor.copyTo(parsedOutput);
      fail();
    } catch (IllegalArgumentException e) {
      // Expected.
    }
  }

  @Test
  public void testCopyToInvalidTypedBuffer() {
    IntBuffer parsedOutput = IntBuffer.allocate(2 * 8 * 8 * 3);
    try {
      tensor.copyTo(parsedOutput);
      fail();
    } catch (IllegalArgumentException e) {
      // Expected.
    }
  }

  @Test
  public void testCopyToWrongType() {
    int[][][][] parsedOutputs = new int[2][8][8][3];
    try {
      tensor.copyTo(parsedOutputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Cannot convert between a TensorFlowLite tensor with type FLOAT32 and a Java object "
                  + "of type [[[[I (which is compatible with the TensorFlowLite type INT32)");
    }
  }

  @Test
  public void testCopyToWrongShape() {
    float[][][][] parsedOutputs = new float[1][8][8][3];
    try {
      tensor.copyTo(parsedOutputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Cannot copy from a TensorFlowLite tensor (output) with shape [2, 8, 8, 3] "
                  + "to a Java object with shape [1, 8, 8, 3].");
    }
  }

  @Test
  public void testSetTo() {
    float[][][][] input = new float[2][8][8][3];
    float[][][][] output = new float[2][8][8][3];

    // Assign from array.
    input[0][0][0][0] = 2.0f;
    tensor.setTo(input);
    tensor.copyTo(output);
    assertThat(output[0][0][0][0]).isEqualTo(2.0f);

    // Assign from direct ByteBuffer.
    ByteBuffer inputByteBuffer =
        ByteBuffer.allocateDirect(2 * 8 * 8 * 3 * 4).order(ByteOrder.nativeOrder());
    inputByteBuffer.putFloat(0, 3.0f);
    tensor.setTo(inputByteBuffer);
    tensor.copyTo(output);
    assertThat(output[0][0][0][0]).isEqualTo(3.0f);

    // Assign from FloatBuffer view of ByteBuffer.
    inputByteBuffer.rewind();
    FloatBuffer inputFloatBuffer = inputByteBuffer.asFloatBuffer();
    inputFloatBuffer.put(0, 5.0f);
    tensor.setTo(inputFloatBuffer);
    tensor.copyTo(output);
    assertThat(output[0][0][0][0]).isEqualTo(5.0f);

    // Assign from (non-direct) FloatBuffer.
    inputFloatBuffer = FloatBuffer.allocate(2 * 8 * 8 * 3);
    inputFloatBuffer.put(0, 5.0f);
    inputFloatBuffer.rewind();
    tensor.setTo(inputFloatBuffer);
    tensor.copyTo(output);
    assertThat(output[0][0][0][0]).isEqualTo(5.0f);

    // Assign from scalar float.
    wrapper.resizeInput(0, new int[0]);
    wrapper.allocateTensors();
    float scalar = 5.0f;
    tensor.setTo(scalar);
    FloatBuffer outputScalar = FloatBuffer.allocate(1);
    tensor.copyTo(outputScalar);
    assertThat(outputScalar.get(0)).isEqualTo(5.0f);

    // Assign from boxed scalar Float.
    Float boxedScalar = 9.0f;
    tensor.setTo(boxedScalar);
    outputScalar = FloatBuffer.allocate(1);
    tensor.copyTo(outputScalar);
    assertThat(outputScalar.get(0)).isEqualTo(9.0f);
  }

  @Test
  public void testSetToInt() {
    wrapper = new NativeInterpreterWrapper(INT_MODEL_PATH);
    tensor = wrapper.getOutputTensor(0);

    int[][][][] input = new int[1][4][4][12];
    int[][][][] output = new int[1][4][4][12];

    // Assign from array.
    input[0][0][0][0] = 2;
    tensor.setTo(input);
    tensor.copyTo(output);
    assertThat(output[0][0][0][0]).isEqualTo(2);

    // Assign from direct ByteBuffer.
    ByteBuffer inputByteBuffer =
        ByteBuffer.allocateDirect(1 * 4 * 4 * 12 * 4).order(ByteOrder.nativeOrder());
    inputByteBuffer.putInt(0, 3);
    tensor.setTo(inputByteBuffer);
    tensor.copyTo(output);
    assertThat(output[0][0][0][0]).isEqualTo(3);

    // Assign from IntBuffer view of ByteBuffer.
    inputByteBuffer.rewind();
    IntBuffer inputIntBuffer = inputByteBuffer.asIntBuffer();
    inputIntBuffer.put(0, 5);
    tensor.setTo(inputIntBuffer);
    tensor.copyTo(output);
    assertThat(output[0][0][0][0]).isEqualTo(5);

    // Assign from (non-direct) IntBuffer.
    inputIntBuffer = IntBuffer.allocate(1 * 4 * 4 * 12);
    inputIntBuffer.put(0, 5);
    tensor.setTo(inputIntBuffer);
    tensor.copyTo(output);
    assertThat(output[0][0][0][0]).isEqualTo(5);
  }

  @Test
  public void testSetToLong() {
    wrapper = new NativeInterpreterWrapper(LONG_MODEL_PATH);
    tensor = wrapper.getOutputTensor(0);

    long[][][][] input = new long[1][4][4][12];
    long[][][][] output = new long[1][4][4][12];

    // Assign from array.
    input[0][0][0][0] = 2;
    tensor.setTo(input);
    tensor.copyTo(output);
    assertThat(output[0][0][0][0]).isEqualTo(2);

    // Assign from direct ByteBuffer.
    ByteBuffer inputByteBuffer =
        ByteBuffer.allocateDirect(1 * 4 * 4 * 12 * 8).order(ByteOrder.nativeOrder());
    inputByteBuffer.putLong(0, 3);
    tensor.setTo(inputByteBuffer);
    tensor.copyTo(output);
    assertThat(output[0][0][0][0]).isEqualTo(3);

    // Assign from LongBuffer view of ByteBuffer.
    inputByteBuffer.rewind();
    LongBuffer inputLongBuffer = inputByteBuffer.asLongBuffer();
    inputLongBuffer.put(0, 5);
    tensor.setTo(inputLongBuffer);
    tensor.copyTo(output);
    assertThat(output[0][0][0][0]).isEqualTo(5);

    // Assign from (non-direct) LongBuffer.
    inputLongBuffer = LongBuffer.allocate(1 * 4 * 4 * 12);
    inputLongBuffer.put(0, 5);
    tensor.setTo(inputLongBuffer);
    tensor.copyTo(output);
    assertThat(output[0][0][0][0]).isEqualTo(5);
  }

  @Test
  public void testSetToNull() {
    try {
      tensor.setTo(null);
      fail();
    } catch (IllegalArgumentException e) {
      // Success.
    }
  }

  @Test
  public void testSetToFloatBuffer() {
    float[] input = new float[2 * 8 * 8 * 3];
    float[] output = new float[2 * 8 * 8 * 3];
    FloatBuffer inputFloatBuffer = FloatBuffer.wrap(input);
    FloatBuffer outputFloatBuffer = FloatBuffer.wrap(output);

    input[0] = 2.0f;
    input[2 * 8 * 8 * 3 - 1] = 7.0f;
    tensor.setTo(inputFloatBuffer);
    tensor.copyTo(outputFloatBuffer);
    assertThat(output[0]).isEqualTo(2.0f);
    assertThat(output[2 * 8 * 8 * 3 - 1]).isEqualTo(7.0f);
  }

  @Test
  public void testSetToInvalidBuffer() {
    Buffer[] inputs = {
      ByteBuffer.allocateDirect(3 * 4).order(ByteOrder.nativeOrder()),
      FloatBuffer.allocate(3),
      IntBuffer.allocate(3),
      LongBuffer.allocate(3)
    };
    for (Buffer input : inputs) {
      try {
        tensor.setTo(input);
        fail();
      } catch (IllegalArgumentException e) {
        // Success.
      }
    }
  }

  @Test
  public void testGetInputShapeIfDifferent() {
    ByteBuffer bytBufferInput = ByteBuffer.allocateDirect(3 * 4).order(ByteOrder.nativeOrder());
    assertThat(tensor.getInputShapeIfDifferent(bytBufferInput)).isNull();

    float[][][][] sameShapeInput = new float[2][8][8][3];
    assertThat(tensor.getInputShapeIfDifferent(sameShapeInput)).isNull();

    float[][][][] differentShapeInput = new float[1][8][8][3];
    assertThat(tensor.getInputShapeIfDifferent(differentShapeInput))
        .isEqualTo(new int[] {1, 8, 8, 3});

    Float differentShapeInputScalar = 5.0f;
    assertThat(tensor.getInputShapeIfDifferent(differentShapeInputScalar)).isEqualTo(new int[] {});
  }

  @Test
  public void testDataTypeOf() {
    float[] testEmptyArray = {};
    DataType dataType = tensor.dataTypeOf(testEmptyArray);
    assertThat(dataType).isEqualTo(DataType.FLOAT32);
    float[] testFloatArray = {0.783f, 0.251f};
    dataType = tensor.dataTypeOf(testFloatArray);
    assertThat(dataType).isEqualTo(DataType.FLOAT32);
    float[][] testMultiDimArray = {testFloatArray, testFloatArray, testFloatArray};
    dataType = tensor.dataTypeOf(testMultiDimArray);
    assertThat(dataType).isEqualTo(DataType.FLOAT32);
    FloatBuffer testFloatBuffer = FloatBuffer.allocate(1);
    dataType = tensor.dataTypeOf(testFloatBuffer);
    assertThat(dataType).isEqualTo(DataType.FLOAT32);
    float testFloat = 1.0f;
    dataType = tensor.dataTypeOf(testFloat);
    assertThat(dataType).isEqualTo(DataType.FLOAT32);
    try {
      double[] testDoubleArray = {0.783, 0.251};
      tensor.dataTypeOf(testDoubleArray);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("cannot resolve DataType of");
    }
    try {
      Float[] testBoxedArray = {0.783f, 0.251f};
      tensor.dataTypeOf(testBoxedArray);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("cannot resolve DataType of [Ljava.lang.Float;");
    }
  }

  @Test
  public void testNumDimensions() {
    int scalar = 1;
    assertThat(TensorImpl.computeNumDimensions(scalar)).isEqualTo(0);
    int[][] array = {{2, 4}, {1, 9}};
    assertThat(TensorImpl.computeNumDimensions(array)).isEqualTo(2);
    try {
      int[] emptyArray = {};
      TensorImpl.computeNumDimensions(emptyArray);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("Array lengths cannot be 0.");
    }
  }

  @Test
  public void testNumElements() {
    int[] scalarShape = {};
    assertThat(TensorImpl.computeNumElements(scalarShape)).isEqualTo(1);
    int[] vectorShape = {3};
    assertThat(TensorImpl.computeNumElements(vectorShape)).isEqualTo(3);
    int[] matrixShape = {3, 4};
    assertThat(TensorImpl.computeNumElements(matrixShape)).isEqualTo(12);
    int[] degenerateShape = {3, 4, 0};
    assertThat(TensorImpl.computeNumElements(degenerateShape)).isEqualTo(0);
  }

  @Test
  public void testFillShape() {
    int[][][] array = {{{23}, {14}, {87}}, {{12}, {42}, {31}}};
    int num = TensorImpl.computeNumDimensions(array);
    int[] shape = new int[num];
    TensorImpl.fillShape(array, 0, shape);
    assertThat(num).isEqualTo(3);
    assertThat(shape[0]).isEqualTo(2);
    assertThat(shape[1]).isEqualTo(3);
    assertThat(shape[2]).isEqualTo(1);
  }

  @Test
  public void testCopyToScalarUnsupported() {
    wrapper.resizeInput(0, new int[0]);
    wrapper.allocateTensors();
    tensor.setTo(5.0f);
    Float outputScalar = 7.0f;
    try {
      tensor.copyTo(outputScalar);
      fail();
    } catch (IllegalArgumentException e) {
      // Expected failure.
    }
  }

  @Test
  public void testUseAfterClose() {
    tensor.close();
    try {
      tensor.numBytes();
      fail();
    } catch (IllegalArgumentException e) {
      // Expected failure.
    }
  }

  @Test
  public void testQuantizationParameters_floatModel() {
    QuantizationParams quantizationParams = tensor.quantizationParams();
    float scale = quantizationParams.getScale();
    long zeroPoint = quantizationParams.getZeroPoint();

    assertThat(scale).isWithin(1e-6f).of(0.0f);
    assertThat(zeroPoint).isEqualTo(0);
  }

  @Test
  public void testQuantizationParameters_quantizedModel() {
    wrapper = new NativeInterpreterWrapper(QUANTIZED_MODEL_PATH);
    tensor = wrapper.getOutputTensor(0);

    QuantizationParams quantizationParams = tensor.quantizationParams();
    float scale = quantizationParams.getScale();
    long zeroPoint = quantizationParams.getZeroPoint();

    assertThat(scale).isWithin(1e-6f).of(0.25f);
    assertThat(zeroPoint).isEqualTo(127);
  }

  @Test
  public void testByteArrayStringTensorInput() {
    NativeInterpreterWrapper wrapper = new NativeInterpreterWrapper(STRING_MODEL_PATH);
    // Test input of string[1]
    wrapper.resizeInput(0, new int[] {1});
    TensorImpl stringTensor = wrapper.getInputTensor(0);
    byte[][] bytes1DStringData = new byte[][] {{0x00, 0x01, 0x02, 0x03}};
    stringTensor.setTo(bytes1DStringData);

    byte[][] byteArray = new byte[][] {new byte[1]};
    assertThat(stringTensor.dataTypeOf(byteArray)).isEqualTo(DataType.STRING);
    assertThat(stringTensor.shape()).isEqualTo(new int[] {1});

    // Test input of scalar string
    wrapper.resizeInput(0, new int[] {});
    byte[] bytesStringData = new byte[] {0x00, 0x01, 0x02, 0x03};
    stringTensor.setTo(bytesStringData);
    assertThat(stringTensor.shape()).isEqualTo(new int[] {});
  }
}
