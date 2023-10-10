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
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;
import org.tensorflow.lite.SupportedFeatures;
import org.tensorflow.lite.TestInit;
import org.tensorflow.lite.TestUtils;

/** Unit tests for {@link org.tensorflow.lite.nnapi.NnApiDelegate}. */
@RunWith(JUnit4.class)
public final class NnApiDelegateTest {

  private static final String MODEL_PATH = "tensorflow/lite/java/src/testdata/add.bin";
  private static final ByteBuffer MODEL_BUFFER = TestUtils.getTestFileAsBuffer(MODEL_PATH);

  private static final Interpreter.Options INTERPRETER_OPTIONS =
      new Interpreter.Options().setRuntime(TfLiteRuntime.PREFER_SYSTEM_OVER_APPLICATION);

  @Before
  public void setUp() throws Exception {
    TestInit.init();
  }

  @Test
  public void testBasic() throws Exception {
    Interpreter.Options options = new Interpreter.Options(INTERPRETER_OPTIONS);
    try (NnApiDelegate delegate = new NnApiDelegate(); // Without options.
        Interpreter interpreter = new Interpreter(MODEL_BUFFER, options.addDelegate(delegate))) {
      assertThat(delegate.getNativeHandle()).isNotEqualTo(0);
    }
  }

  @Test
  public void testAccessBeforeInterpreterInitialized() throws Exception {
    try (NnApiDelegate delegate = new NnApiDelegate()) {
      try {
        delegate.getNativeHandle();
        fail("Expected IllegalStateException to be thrown");
      } catch (IllegalStateException e) {
        assertThat(e)
            .hasMessageThat()
            .contains("Should not access delegate before interpreter has been constructed");
      }
      // Merely adding the delegate to the options isn't enough either.
      Interpreter.Options options =
          new Interpreter.Options(INTERPRETER_OPTIONS).addDelegate(delegate);
      try {
        delegate.getNativeHandle();
        fail("Expected IllegalStateException to be thrown");
      } catch (IllegalStateException e) {
        assertThat(e)
            .hasMessageThat()
            .contains("Should not access delegate before interpreter has been constructed");
      }
    }
  }

  @Test
  public void testWithoutNnApiDelegateOptions() throws Exception {
    Interpreter.Options options = new Interpreter.Options(INTERPRETER_OPTIONS);
    try (NnApiDelegate delegate = new NnApiDelegate(); // Without options.
        Interpreter interpreter = new Interpreter(MODEL_BUFFER, options.addDelegate(delegate))) {
      assertThat(delegate.getNativeHandle()).isNotEqualTo(0);
    } catch (IllegalStateException e) {
      // This can happen if the TF Lite runtime isn't linked into the test.
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Couldn't find TensorFlow Lite runtime's InterpreterFactoryImpl class -- make sure"
                  + " your app links in the right TensorFlow Lite runtime. You should declare a"
                  + " build dependency on org.tensorflow.lite:tensorflow-lite, or call"
                  + " setTfLiteRuntime with a value other than TfLiteRuntime.FROM_APPLICATION_ONLY"
                  + " (see docs for"
                  + " org.tensorflow.lite.nnapi.NnApiDelegate.Options#setTfLiteRuntime).");
    }
  }

  @Test
  public void testInterpreterWithNnApi() throws Exception {
    Interpreter.Options options = new Interpreter.Options(INTERPRETER_OPTIONS);
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
    if (!SupportedFeatures.supportsXnnpack()) {
      System.err.println("Not testing NNAPI with XNNPACK, since XNNPACK isn't supported.");
      return;
    }
    Interpreter.Options options = new Interpreter.Options(INTERPRETER_OPTIONS);
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
    NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
    Interpreter.Options options = new Interpreter.Options(INTERPRETER_OPTIONS);
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
    Interpreter.Options options = new Interpreter.Options(INTERPRETER_OPTIONS);
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
  public void testGetNnApiErrnoBeforeInitializingInterpreterReturnsZero() {
    NnApiDelegate delegate = new NnApiDelegate();
    assertThat(delegate.getNnapiErrno()).isEqualTo(0);
  }

  @Test
  public void testGetNnApiErrnoThrowsExceptionAfterClosingDelegate() {
    NnApiDelegate delegate = new NnApiDelegate();
    Interpreter interpreter =
        new Interpreter(
            MODEL_BUFFER, new Interpreter.Options(INTERPRETER_OPTIONS).addDelegate(delegate));
    assertThat(delegate.getNnapiErrno()).isEqualTo(0);

    delegate.close();
    try {
      delegate.getNnapiErrno();
      fail("Expected IllegalStateException to be thrown.");
    } catch (IllegalStateException expected) {
      assertThat(expected)
          .hasMessageThat()
          .contains("Should not access delegate after delegate has been closed");
    }
  }

  @Test
  public void testSupportLibraryIsSetFromHandle() {
    NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
    Interpreter.Options options = new Interpreter.Options(INTERPRETER_OPTIONS);
    long mockSlHandle = getMockSlHandle();
    try (NnApiDelegate delegate =
            new NnApiDelegate(
                nnApiOptions.setNnApiSupportLibraryHandle(mockSlHandle).setUseNnapiCpu(true));
        Interpreter interpreter =
            new Interpreter(MODEL_BUFFER, options.addDelegate(delegate).setUseXNNPACK(false))) {
      float[] oneD = {1.23f, 6.54f, 7.81f};
      float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
      float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
      float[][][][] fourD = {threeD, threeD};
      float[][][][] parsedOutputs = new float[2][8][8][3];
      interpreter.run(fourD, parsedOutputs);
      // Not checking outputs since we're using mock NNAPI support library
      assertThat(hasNnApiSlBeenCalled()).isTrue();
    }
    closeMockSl(mockSlHandle);
  }

  /**
   * Allocates a mock NNAPI Support Library object. The mock library does nothing but remembers if
   * any of its functions were called at least once.
   */
  private static native long getMockSlHandle();

  /** Returns true if any function from the mock Support Library has been called at least once. */
  private static native boolean hasNnApiSlBeenCalled();

  /** Deallocates the Support Library object. */
  private static native void closeMockSl(long handle);
}
