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

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.lite.Interpreter}. */
@RunWith(JUnit4.class)
public final class InterpreterTest {

  private static final String MODEL_PATH = "tensorflow/lite/java/src/testdata/add.bin";
  private static final String MULTIPLE_INPUTS_MODEL_PATH =
      "tensorflow/lite/testdata/multi_add.bin";
  private static final String FLEX_MODEL_PATH =
      "tensorflow/lite/testdata/multi_add_flex.bin";
  private static final String UNKNOWN_DIMS_MODEL_PATH =
      "tensorflow/lite/java/src/testdata/add_unknown_dimensions.bin";
  private static final String DYNAMIC_SHAPES_MODEL_PATH =
      "tensorflow/lite/testdata/dynamic_shapes.bin";

  private static final ByteBuffer MODEL_BUFFER = TestUtils.getTestFileAsBuffer(MODEL_PATH);
  private static final ByteBuffer MULTIPLE_INPUTS_MODEL_BUFFER =
      TestUtils.getTestFileAsBuffer(MULTIPLE_INPUTS_MODEL_PATH);
  private static final ByteBuffer FLEX_MODEL_BUFFER =
      TestUtils.getTestFileAsBuffer(FLEX_MODEL_PATH);
  private static final ByteBuffer UNKNOWN_DIMS_MODEL_PATH_BUFFER =
      TestUtils.getTestFileAsBuffer(UNKNOWN_DIMS_MODEL_PATH);
  private static final ByteBuffer DYNAMIC_SHAPES_MODEL_BUFFER =
      TestUtils.getTestFileAsBuffer(DYNAMIC_SHAPES_MODEL_PATH);

  @Test
  public void testInterpreter() throws Exception {
    Interpreter interpreter = new Interpreter(MODEL_BUFFER);
    assertThat(interpreter).isNotNull();
    assertThat(interpreter.getInputTensorCount()).isEqualTo(1);
    assertThat(interpreter.getInputTensor(0).dataType()).isEqualTo(DataType.FLOAT32);
    assertThat(interpreter.getOutputTensorCount()).isEqualTo(1);
    assertThat(interpreter.getOutputTensor(0).dataType()).isEqualTo(DataType.FLOAT32);
    interpreter.close();
  }

  @Test
  public void testInterpreterWithOptions() throws Exception {
    Interpreter interpreter =
        new Interpreter(
            MODEL_BUFFER,
            new Interpreter.Options()
                .setNumThreads(2)
                .setUseNNAPI(true)
                .setAllowFp16PrecisionForFp32(false)
                .setAllowBufferHandleOutput(false));
    assertThat(interpreter).isNotNull();
    assertThat(interpreter.getInputTensorCount()).isEqualTo(1);
    assertThat(interpreter.getInputTensor(0).dataType()).isEqualTo(DataType.FLOAT32);
    assertThat(interpreter.getOutputTensorCount()).isEqualTo(1);
    assertThat(interpreter.getOutputTensor(0).dataType()).isEqualTo(DataType.FLOAT32);
    interpreter.close();
  }

  @Test
  public void testRunWithFileModel() throws Exception {
    if (!TestUtils.supportsFilePaths()) {
      return;
    }
    Interpreter interpreter = new Interpreter(new File(MODEL_PATH));
    float[] oneD = {1.23f, 6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    float[][][][] fourD = {threeD, threeD};
    float[][][][] parsedOutputs = new float[2][8][8][3];
    interpreter.run(fourD, parsedOutputs);
    float[] outputOneD = parsedOutputs[0][0][0];
    float[] expected = {3.69f, 19.62f, 23.43f};
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
    interpreter.close();
  }

  @Test
  public void testRunWithDirectByteBufferModel() throws Exception {
    ByteBuffer byteBuffer = ByteBuffer.allocateDirect(MODEL_BUFFER.capacity());
    byteBuffer.order(ByteOrder.nativeOrder());
    byteBuffer.put(MODEL_BUFFER);
    Interpreter interpreter = new Interpreter(byteBuffer);
    float[] oneD = {1.23f, 6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    float[][][][] fourD = {threeD, threeD};
    float[][][][] parsedOutputs = new float[2][8][8][3];
    interpreter.run(fourD, parsedOutputs);
    float[] outputOneD = parsedOutputs[0][0][0];
    float[] expected = {3.69f, 19.62f, 23.43f};
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
    interpreter.close();
  }

  @Test
  public void testRunWithInvalidByteBufferModel() throws Exception {
    ByteBuffer byteBuffer = ByteBuffer.allocate(MODEL_BUFFER.capacity());
    byteBuffer.order(ByteOrder.nativeOrder());
    byteBuffer.put(MODEL_BUFFER);
    try {
      new Interpreter(byteBuffer);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Model ByteBuffer should be either a MappedByteBuffer"
                  + " of the model file, or a direct ByteBuffer using ByteOrder.nativeOrder()");
    }
  }

  @Test
  public void testRun() {
    Interpreter interpreter = new Interpreter(MODEL_BUFFER);
    Float[] oneD = {1.23f, 6.54f, 7.81f};
    Float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    Float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    Float[][][][] fourD = {threeD, threeD};
    Float[][][][] parsedOutputs = new Float[2][8][8][3];
    try {
      interpreter.run(fourD, parsedOutputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("cannot resolve DataType of [[[[Ljava.lang.Float;");
    }
    interpreter.close();
  }

  @Test
  public void testRunWithBoxedInputs() {
    Interpreter interpreter = new Interpreter(MODEL_BUFFER);
    float[] oneD = {1.23f, 6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    float[][][][] fourD = {threeD, threeD};
    float[][][][] parsedOutputs = new float[2][8][8][3];
    interpreter.run(fourD, parsedOutputs);
    float[] outputOneD = parsedOutputs[0][0][0];
    float[] expected = {3.69f, 19.62f, 23.43f};
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
    interpreter.close();
  }

  @Test
  public void testRunForMultipleInputsOutputs() {
    Interpreter interpreter = new Interpreter(MULTIPLE_INPUTS_MODEL_BUFFER);
    assertThat(interpreter.getInputTensorCount()).isEqualTo(4);
    assertThat(interpreter.getInputTensor(0).index()).isGreaterThan(-1);
    assertThat(interpreter.getInputTensor(0).dataType()).isEqualTo(DataType.FLOAT32);
    assertThat(interpreter.getInputTensor(1).dataType()).isEqualTo(DataType.FLOAT32);
    assertThat(interpreter.getInputTensor(2).dataType()).isEqualTo(DataType.FLOAT32);
    assertThat(interpreter.getInputTensor(3).dataType()).isEqualTo(DataType.FLOAT32);
    assertThat(interpreter.getOutputTensorCount()).isEqualTo(2);
    assertThat(interpreter.getOutputTensor(0).index()).isGreaterThan(-1);
    assertThat(interpreter.getOutputTensor(0).dataType()).isEqualTo(DataType.FLOAT32);
    assertThat(interpreter.getOutputTensor(1).dataType()).isEqualTo(DataType.FLOAT32);

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

  @Test
  public void testRunWithByteBufferOutput() {
    float[] oneD = {1.23f, 6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    float[][][][] fourD = {threeD, threeD};
    ByteBuffer parsedOutput =
        ByteBuffer.allocateDirect(2 * 8 * 8 * 3 * 4).order(ByteOrder.nativeOrder());
    try (Interpreter interpreter = new Interpreter(MODEL_BUFFER)) {
      interpreter.run(fourD, parsedOutput);
    }
    float[] outputOneD = {
      parsedOutput.getFloat(0), parsedOutput.getFloat(4), parsedOutput.getFloat(8)
    };
    float[] expected = {3.69f, 19.62f, 23.43f};
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
  }

  @Test
  public void testRunWithScalarInput() {
    FloatBuffer parsedOutput = FloatBuffer.allocate(1);
    try (Interpreter interpreter = new Interpreter(MODEL_BUFFER)) {
      interpreter.run(2.37f, parsedOutput);
    }
    assertThat(parsedOutput.get(0)).isWithin(0.1f).of(7.11f);
  }

  @Test
  public void testResizeInput() {
    try (Interpreter interpreter = new Interpreter(MODEL_BUFFER)) {
      int[] inputDims = {1};
      interpreter.resizeInput(0, inputDims);
      assertThat(interpreter.getInputTensor(0).shape()).isEqualTo(inputDims);
      ByteBuffer input = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder());
      ByteBuffer output = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder());
      interpreter.run(input, output);
      assertThat(interpreter.getOutputTensor(0).shape()).isEqualTo(inputDims);
    }
  }

  @Test
  public void testAllocateTensors() {
    try (Interpreter interpreter = new Interpreter(MODEL_BUFFER)) {
      // Redundant allocateTensors() should have no effect.
      interpreter.allocateTensors();

      // allocateTensors() should propagate resizes.
      int[] inputDims = {1};
      assertThat(interpreter.getOutputTensor(0).shape()).isNotEqualTo(inputDims);
      interpreter.resizeInput(0, inputDims);
      assertThat(interpreter.getOutputTensor(0).shape()).isNotEqualTo(inputDims);
      interpreter.allocateTensors();
      assertThat(interpreter.getOutputTensor(0).shape()).isEqualTo(inputDims);

      // Additional redundant calls should have no effect.
      interpreter.allocateTensors();
      assertThat(interpreter.getOutputTensor(0).shape()).isEqualTo(inputDims);

      // Execution should succeed as expected.
      ByteBuffer input = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder());
      ByteBuffer output = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder());
      interpreter.run(input, output);
      assertThat(interpreter.getOutputTensor(0).shape()).isEqualTo(inputDims);
    }
  }

  @Test
  public void testUnknownDims() {
    try (Interpreter interpreter = new Interpreter(UNKNOWN_DIMS_MODEL_PATH_BUFFER)) {
      int[] inputDims = {1, 1, 3, 3};
      int[] inputDimsSignature = {1, -1, 3, 3};
      assertThat(interpreter.getInputTensor(0).shape()).isEqualTo(inputDims);
      assertThat(interpreter.getInputTensor(0).shapeSignature()).isEqualTo(inputDimsSignature);

      // Resize tensor with strict checking. Try invalid resize.
      inputDims[2] = 5;
      try {
        interpreter.resizeInput(0, inputDims, true);
        fail();
      } catch (IllegalArgumentException e) {
        assertThat(e)
            .hasMessageThat()
            .contains(
                "ResizeInputTensorStrict only allows mutating unknown dimensions identified by -1");
      }
      inputDims[2] = 3;

      // Set the dimension of the unknown dimension to the expected dimension and ensure shape
      // signature doesn't change.
      inputDims[1] = 3;
      interpreter.resizeInput(0, inputDims, true);
      assertThat(interpreter.getInputTensor(0).shape()).isEqualTo(inputDims);
      assertThat(interpreter.getInputTensor(0).shapeSignature()).isEqualTo(inputDimsSignature);

      ByteBuffer input =
          ByteBuffer.allocateDirect(1 * 3 * 3 * 3 * 4).order(ByteOrder.nativeOrder());
      ByteBuffer output =
          ByteBuffer.allocateDirect(1 * 3 * 3 * 3 * 4).order(ByteOrder.nativeOrder());
      interpreter.run(input, output);
      assertThat(interpreter.getOutputTensor(0).shape()).isEqualTo(inputDims);
    }
  }

  @Test
  public void testRunWithWrongInputType() {
    Interpreter interpreter = new Interpreter(MODEL_BUFFER);
    int[] oneD = {4, 3, 9};
    int[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    int[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    int[][][][] fourD = {threeD, threeD};
    float[][][][] parsedOutputs = new float[2][8][8][3];
    try {
      interpreter.run(fourD, parsedOutputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Cannot convert between a TensorFlowLite tensor with type "
                  + "FLOAT32 and a Java object of type [[[[I (which is compatible with the"
                  + " TensorFlowLite type INT32)");
    }
    interpreter.close();
  }

  @Test
  public void testRunWithUnsupportedInputType() {
    DoubleBuffer doubleBuffer = DoubleBuffer.allocate(10);
    float[][][][] parsedOutputs = new float[2][8][8][3];
    try (Interpreter interpreter = new Interpreter(MODEL_BUFFER)) {
      interpreter.run(doubleBuffer, parsedOutputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("DataType error: cannot resolve DataType of");
    }
  }

  @Test
  public void testRunWithWrongOutputType() {
    Interpreter interpreter = new Interpreter(MODEL_BUFFER);
    float[] oneD = {1.23f, 6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    float[][][][] fourD = {threeD, threeD};
    int[][][][] parsedOutputs = new int[2][8][8][3];
    try {
      interpreter.run(fourD, parsedOutputs);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Cannot convert between a TensorFlowLite tensor with type "
                  + "FLOAT32 and a Java object of type [[[[I (which is compatible with the"
                  + " TensorFlowLite type INT32)");
    }
    interpreter.close();
  }

  @Test
  public void testGetInputIndex() {
    Interpreter interpreter = new Interpreter(MODEL_BUFFER);
    try {
      interpreter.getInputIndex("WrongInputName");
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "'WrongInputName' is not a valid name for any input. Names of inputs and their "
                  + "indexes are {input=0}");
    }
    int index = interpreter.getInputIndex("input");
    assertThat(index).isEqualTo(0);
  }

  @Test
  public void testGetOutputIndex() {
    Interpreter interpreter = new Interpreter(MODEL_BUFFER);
    try {
      interpreter.getOutputIndex("WrongOutputName");
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "'WrongOutputName' is not a valid name for any output. Names of outputs and their"
                  + " indexes are {output=0}");
    }
    int index = interpreter.getOutputIndex("output");
    assertThat(index).isEqualTo(0);
  }

  @Test
  public void testTurnOnNNAPI() throws Exception {
    Interpreter interpreter =
        new Interpreter(
            MODEL_BUFFER,
            new Interpreter.Options().setUseNNAPI(true).setAllowFp16PrecisionForFp32(true));
    float[] oneD = {1.23f, 6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    float[][][][] fourD = {threeD, threeD};
    float[][][][] parsedOutputs = new float[2][8][8][3];
    interpreter.run(fourD, parsedOutputs);
    float[] outputOneD = parsedOutputs[0][0][0];
    float[] expected = {3.69f, 19.62f, 23.43f};
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
    interpreter.close();
  }

  @Test
  public void testRedundantClose() throws Exception {
    Interpreter interpreter = new Interpreter(MODEL_BUFFER);
    interpreter.close();
    interpreter.close();
  }

  @Test
  public void testNullInputs() throws Exception {
    Interpreter interpreter = new Interpreter(MODEL_BUFFER);
    try {
      interpreter.run(null, new float[2][8][8][3]);
      fail();
    } catch (IllegalArgumentException e) {
      // Expected failure.
    }
    interpreter.close();
  }

  @Test
  public void testNullOutputs() throws Exception {
    Interpreter interpreter = new Interpreter(MODEL_BUFFER);
    try {
      interpreter.run(new float[2][8][8][3], null);
      fail();
    } catch (IllegalArgumentException e) {
      // Expected failure.
    }
    interpreter.close();
  }

  // Smoke test validating that flex model loading fails when the flex delegate is not linked.
  @Test
  public void testFlexModel() throws Exception {
    try {
      new Interpreter(FLEX_MODEL_BUFFER);
      fail();
    } catch (IllegalStateException e) {
      // Expected failure.
    }
  }

  @Test
  public void testDelegate() throws Exception {
    System.loadLibrary("tensorflowlite_test_jni");
    Delegate delegate =
        new Delegate() {
          @Override
          public long getNativeHandle() {
            return getNativeHandleForDelegate();
          }
        };
    Interpreter interpreter =
        new Interpreter(MODEL_BUFFER, new Interpreter.Options().addDelegate(delegate));

    // The native delegate stubs out the graph with a single op that produces the scalar value 7.
    float[] oneD = {1.23f, 6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    float[][][][] fourD = {threeD, threeD};
    float[][][][] parsedOutputs = new float[2][8][8][3];
    interpreter.run(fourD, parsedOutputs);
    float[] outputOneD = parsedOutputs[0][0][0];
    float[] expected = {7.0f, 7.0f, 7.0f};
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();

    interpreter.close();
  }

  @Test
  public void testNullInputsAndOutputsWithDelegate() throws Exception {
    System.loadLibrary("tensorflowlite_test_jni");
    Delegate delegate =
        new Delegate() {
          @Override
          public long getNativeHandle() {
            return getNativeHandleForDelegate();
          }
        };
    Interpreter interpreter =
        new Interpreter(MODEL_BUFFER, new Interpreter.Options().addDelegate(delegate));
    // The delegate installs a custom buffer handle for all tensors, in turn allowing null to be
    // provided for the inputs/outputs (as the client can reference the buffer directly).
    interpreter.run(new float[2][8][8][3], null);
    interpreter.run(null, new float[2][8][8][3]);
    interpreter.close();
  }

  @Test
  public void testModifyGraphWithDelegate() throws Exception {
    System.loadLibrary("tensorflowlite_test_jni");
    Delegate delegate =
        new Delegate() {
          @Override
          public long getNativeHandle() {
            return getNativeHandleForDelegate();
          }
        };
    Interpreter interpreter = new Interpreter(MODEL_BUFFER);
    interpreter.modifyGraphWithDelegate(delegate);

    // The native delegate stubs out the graph with a single op that produces the scalar value 7.
    float[] oneD = {1.23f, 6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    float[][][][] fourD = {threeD, threeD};
    float[][][][] parsedOutputs = new float[2][8][8][3];
    interpreter.run(fourD, parsedOutputs);
    float[] outputOneD = parsedOutputs[0][0][0];
    float[] expected = {7.0f, 7.0f, 7.0f};
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();

    interpreter.close();
  }

  @Test
  public void testInvalidDelegate() throws Exception {
    System.loadLibrary("tensorflowlite_test_jni");
    Delegate delegate =
        new Delegate() {
          @Override
          public long getNativeHandle() {
            return getNativeHandleForInvalidDelegate();
          }
        };
    try {
      Interpreter interpreter =
          new Interpreter(MODEL_BUFFER, new Interpreter.Options().addDelegate(delegate));
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("Internal error: Failed to apply delegate");
    }
  }

  @Test
  public void testNullDelegate() throws Exception {
    System.loadLibrary("tensorflowlite_test_jni");
    Delegate delegate =
        new Delegate() {
          @Override
          public long getNativeHandle() {
            return 0;
          }
        };
    try {
      Interpreter interpreter =
          new Interpreter(MODEL_BUFFER, new Interpreter.Options().addDelegate(delegate));
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("Internal error: Invalid handle to delegate");
    }
  }

  @Test
  public void testResetVariableTensors() throws Exception {
    float[][][][] inputs = new float[2][8][8][3];
    float[][][][] parsedOutputs = new float[2][8][8][3];

    // Smoke test to ensure resetting variables at various times in a simple graph doesn't fail.
    // TODO(b/138197256): Test with model that has variables.
    try (Interpreter interpreter = new Interpreter(MODEL_BUFFER)) {
      interpreter.resetVariableTensors();
      interpreter.run(inputs, parsedOutputs);

      interpreter.resetVariableTensors();
      interpreter.resetVariableTensors();
      interpreter.run(inputs, parsedOutputs);
    }
  }

  private static FloatBuffer fill(FloatBuffer buffer, float value) {
    while (buffer.hasRemaining()) {
      buffer.put(value);
    }
    buffer.rewind();
    return buffer;
  }

  // Regression test case to ensure that graphs with dynamically computed shapes work properly.
  // Historically, direct ByteBuffer addresses would overwrite the arena-allocated tensor input
  // pointers. Normally this works fine, but for dynamic graphs, the original input tensor pointers
  // may be "restored" at invocation time by the arena allocator, resetting the direct ByteBuffer
  // address and leading to stale input data being used.
  @Test
  public void testDynamicShapesWithDirectBufferInputs() {
    try (Interpreter interpreter = new Interpreter(DYNAMIC_SHAPES_MODEL_BUFFER)) {
      ByteBuffer input0 =
          ByteBuffer.allocateDirect(8 * 42 * 1024 * 4).order(ByteOrder.nativeOrder());
      ByteBuffer input1 =
          ByteBuffer.allocateDirect(1 * 90 * 1024 * 4).order(ByteOrder.nativeOrder());
      ByteBuffer input2 = ByteBuffer.allocateDirect(1 * 4).order(ByteOrder.nativeOrder());
      Object[] inputs = {input0, input1, input2};

      fill(input0.asFloatBuffer(), 2.0f);
      fill(input1.asFloatBuffer(), 0.5f);
      // Note that the value of this input dictates the shape of the output.
      fill(input2.asFloatBuffer(), 1.0f);

      FloatBuffer output = FloatBuffer.allocate(8 * 1 * 1024);
      Map<Integer, Object> outputs = new HashMap<>();
      outputs.put(0, output);

      interpreter.runForMultipleInputsOutputs(inputs, outputs);

      FloatBuffer expected = fill(FloatBuffer.allocate(8 * 1 * 1024), 2.0f);
      assertThat(output.array()).usingTolerance(0.1f).containsExactly(expected.array()).inOrder();
    }
  }

  private static native long getNativeHandleForDelegate();

  private static native long getNativeHandleForInvalidDelegate();
}
