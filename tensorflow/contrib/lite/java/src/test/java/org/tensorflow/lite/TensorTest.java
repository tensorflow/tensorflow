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
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.lite.Tensor}. */
@RunWith(JUnit4.class)
public final class TensorTest {

  private static final String MODEL_PATH =
      "tensorflow/contrib/lite/java/src/testdata/add.bin";

  private NativeInterpreterWrapper wrapper;
  private Tensor tensor;

  @Before
  public void setUp() {
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
    assertThat(tensor.dataType()).isEqualTo(DataType.FLOAT32);
    assertThat(tensor.numBytes()).isEqualTo(2 * 8 * 8 * 3 * 4);
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
              "Cannot copy between a TensorFlowLite tensor with shape [2, 8, 8, 3] "
                  + "and a Java object with shape [1, 8, 8, 3].");
    }
  }

  @Test
  public void testSetTo() {
    float[][][][] input = new float[2][8][8][3];
    float[][][][] output = new float[2][8][8][3];
    ByteBuffer inputByteBuffer =
        ByteBuffer.allocateDirect(2 * 8 * 8 * 3 * 4).order(ByteOrder.nativeOrder());

    input[0][0][0][0] = 2.0f;
    tensor.setTo(input);
    tensor.copyTo(output);
    assertThat(output[0][0][0][0]).isEqualTo(2.0f);

    inputByteBuffer.putFloat(0, 3.0f);
    tensor.setTo(inputByteBuffer);
    tensor.copyTo(output);
    assertThat(output[0][0][0][0]).isEqualTo(3.0f);
  }

  @Test
  public void testSetToInvalidByteBuffer() {
    ByteBuffer input = ByteBuffer.allocateDirect(3 * 4).order(ByteOrder.nativeOrder());
    try {
      tensor.setTo(input);
      fail();
    } catch (IllegalArgumentException e) {
      // Success.
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
  }

  @Test
  public void testDataTypeOf() {
    float[] testEmptyArray = {};
    DataType dataType = Tensor.dataTypeOf(testEmptyArray);
    assertThat(dataType).isEqualTo(DataType.FLOAT32);
    float[] testFloatArray = {0.783f, 0.251f};
    dataType = Tensor.dataTypeOf(testFloatArray);
    assertThat(dataType).isEqualTo(DataType.FLOAT32);
    float[][] testMultiDimArray = {testFloatArray, testFloatArray, testFloatArray};
    dataType = Tensor.dataTypeOf(testFloatArray);
    assertThat(dataType).isEqualTo(DataType.FLOAT32);
    try {
      double[] testDoubleArray = {0.783, 0.251};
      Tensor.dataTypeOf(testDoubleArray);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("cannot resolve DataType of");
    }
    try {
      Float[] testBoxedArray = {0.783f, 0.251f};
      Tensor.dataTypeOf(testBoxedArray);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("cannot resolve DataType of [Ljava.lang.Float;");
    }
  }

  @Test
  public void testNumDimensions() {
    int scalar = 1;
    assertThat(Tensor.numDimensions(scalar)).isEqualTo(0);
    int[][] array = {{2, 4}, {1, 9}};
    assertThat(Tensor.numDimensions(array)).isEqualTo(2);
    try {
      int[] emptyArray = {};
      Tensor.numDimensions(emptyArray);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("Array lengths cannot be 0.");
    }
  }

  @Test
  public void testFillShape() {
    int[][][] array = {{{23}, {14}, {87}}, {{12}, {42}, {31}}};
    int num = Tensor.numDimensions(array);
    int[] shape = new int[num];
    Tensor.fillShape(array, 0, shape);
    assertThat(num).isEqualTo(3);
    assertThat(shape[0]).isEqualTo(2);
    assertThat(shape[1]).isEqualTo(3);
    assertThat(shape[2]).isEqualTo(1);
  }
}
