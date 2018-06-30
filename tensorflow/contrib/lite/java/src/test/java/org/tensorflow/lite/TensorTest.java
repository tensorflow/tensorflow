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
  private long nativeHandle;

  @Before
  public void setUp() {
    wrapper = new NativeInterpreterWrapper(MODEL_PATH);
    float[] oneD = {1.23f, 6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    float[][][][] fourD = {threeD, threeD};
    Object[] inputs = {fourD};
    Tensor[] outputs = wrapper.run(inputs);
    nativeHandle = outputs[0].nativeHandle;
  }

  @After
  public void tearDown() {
    wrapper.close();
  }

  @Test
  public void testFromHandle() throws Exception {
    Tensor tensor = Tensor.fromHandle(nativeHandle);
    assertThat(tensor).isNotNull();
    int[] expectedShape = {2, 8, 8, 3};
    assertThat(tensor.shapeCopy).isEqualTo(expectedShape);
    assertThat(tensor.dtype).isEqualTo(DataType.FLOAT32);
  }

  @Test
  public void testCopyTo() {
    Tensor tensor = Tensor.fromHandle(nativeHandle);
    float[][][][] parsedOutputs = new float[2][8][8][3];
    tensor.copyTo(parsedOutputs);
    float[] outputOneD = parsedOutputs[0][0][0];
    float[] expected = {3.69f, 19.62f, 23.43f};
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
  }

  @Test
  public void testCopyToWrongType() {
    Tensor tensor = Tensor.fromHandle(nativeHandle);
    int[][][][] parsedOutputs = new int[2][8][8][3];
    try {
      tensor.copyTo(parsedOutputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Cannot convert an TensorFlowLite tensor with type "
                  + "FLOAT32 to a Java object of type [[[[I (which is compatible with the TensorFlowLite "
                  + "type INT32)");
    }
  }

  @Test
  public void testCopyToWrongShape() {
    Tensor tensor = Tensor.fromHandle(nativeHandle);
    float[][][][] parsedOutputs = new float[1][8][8][3];
    try {
      tensor.copyTo(parsedOutputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Shape of output target [1, 8, 8, 3] does not match "
                  + "with the shape of the Tensor [2, 8, 8, 3].");
    }
  }
}
