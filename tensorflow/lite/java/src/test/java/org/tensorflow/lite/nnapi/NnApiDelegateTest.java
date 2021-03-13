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

package org.tensorflow.lite.nnapi;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import java.nio.ByteBuffer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.TestUtils;

/** Unit tests for {@link org.tensorflow.lite.nnapi.NnApiDelegate}. */
@RunWith(JUnit4.class)
public final class NnApiDelegateTest {

  private static final String MODEL_PATH = "tensorflow/lite/java/src/testdata/add.bin";
  private static final ByteBuffer MODEL_BUFFER = TestUtils.getTestFileAsBuffer(MODEL_PATH);

  @Test
  public void testBasic() throws Exception {
    try (NnApiDelegate delegate = new NnApiDelegate()) {
      assertThat(delegate.getNativeHandle()).isNotEqualTo(0);
    }
  }

  @Test
  public void testInterpreterWithNnApi() throws Exception {
    Interpreter.Options options = new Interpreter.Options();
    try (NnApiDelegate delegate = new NnApiDelegate();
        Interpreter interpreter = new Interpreter(MODEL_BUFFER, options.addDelegate(delegate))) {
      float[] oneD = {1.23f, 6.54f, 7.81f};
      float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
      float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
      float[][][][] fourD = {threeD, threeD};
      float[][][][] parsedOutputs = new float[2][8][8][3];
      interpreter.run(fourD, parsedOutputs);
      float[] outputOneD = parsedOutputs[0][0][0];
      float[] expected = {3.69f, 19.62f, 23.43f};
      assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
    }
  }

  @Test
  public void testInterpreterWithNnApiAndXNNPack() throws Exception {
    Interpreter.Options options = new Interpreter.Options();
    options.setUseXNNPACK(true);

    try (NnApiDelegate delegate = new NnApiDelegate();
        Interpreter interpreter = new Interpreter(MODEL_BUFFER, options.addDelegate(delegate))) {
      float[] oneD = {1.23f, 6.54f, 7.81f};
      float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
      float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
      float[][][][] fourD = {threeD, threeD};
      float[][][][] parsedOutputs = new float[2][8][8][3];
      interpreter.run(fourD, parsedOutputs);
      float[] outputOneD = parsedOutputs[0][0][0];
      float[] expected = {3.69f, 19.62f, 23.43f};
      assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
    }
  }

  @Test
  public void testInterpreterWithNnApiAllowFp16() throws Exception {
    Interpreter.Options options = new Interpreter.Options();
    NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
    nnApiOptions.setAllowFp16(true);

    try (NnApiDelegate delegate = new NnApiDelegate(nnApiOptions);
        Interpreter interpreter = new Interpreter(MODEL_BUFFER, options.addDelegate(delegate))) {
      float[] oneD = {1.23f, 6.54f, 7.81f};
      float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
      float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
      float[][][][] fourD = {threeD, threeD};
      float[][][][] parsedOutputs = new float[2][8][8][3];
      interpreter.run(fourD, parsedOutputs);
      float[] outputOneD = parsedOutputs[0][0][0];
      float[] expected = {3.69f, 19.62f, 23.43f};
      assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
    }
  }

  @Test
  public void testGetNnApiErrnoReturnsZeroIfNoNnapiCallFailed() throws Exception {
    Interpreter.Options options = new Interpreter.Options();
    try (NnApiDelegate delegate = new NnApiDelegate();
        Interpreter interpreter = new Interpreter(MODEL_BUFFER, options.addDelegate(delegate))) {
      float[] oneD = {1.23f, 6.54f, 7.81f};
      float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
      float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
      float[][][][] fourD = {threeD, threeD};
      float[][][][] parsedOutputs = new float[2][8][8][3];
      interpreter.run(fourD, parsedOutputs);

      assertThat(delegate.getNnapiErrno()).isEqualTo(0);
      assertThat(delegate.hasErrors()).isFalse();
    }
  }

  @Test
  public void testGetNnApiErrnoThrowsExceptionAfterClosingDelegate() {
    NnApiDelegate delegate = new NnApiDelegate();
    assertThat(delegate.getNnapiErrno()).isEqualTo(0);

    delegate.close();
    try {
      delegate.getNnapiErrno();
      fail("Expected IllegalStateException to be thrown.");
    } catch (IllegalStateException expected) {
    }
  }
}
