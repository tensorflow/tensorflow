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
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.lite.Interpreter}. */
@RunWith(JUnit4.class)
public final class InterpreterTest {

  private static final File MODEL_FILE =
      new File("tensorflow/contrib/lite/java/src/testdata/add.bin");

  private static final File MOBILENET_MODEL_FILE =
      new File("tensorflow/contrib/lite/java/src/testdata/mobilenet.tflite.bin");

  @Test
  public void testInterpreter() throws Exception {
    Interpreter interpreter = new Interpreter(MODEL_FILE);
    assertThat(interpreter).isNotNull();
    interpreter.close();
  }

  @Test
  public void testRunWithMappedByteBufferModel() throws Exception {
    Path path = MODEL_FILE.toPath();
    FileChannel fileChannel =
        (FileChannel) Files.newByteChannel(path, EnumSet.of(StandardOpenOption.READ));
    MappedByteBuffer mappedByteBuffer =
        fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size());
    Interpreter interpreter = new Interpreter(mappedByteBuffer);
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
    fileChannel.close();
  }

  @Test
  public void testRun() {
    Interpreter interpreter = new Interpreter(MODEL_FILE);
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
    Interpreter interpreter = new Interpreter(MODEL_FILE);
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
    Interpreter interpreter = new Interpreter(MODEL_FILE);
    float[] oneD = {1.23f, 6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    float[][][][] fourD = {threeD, threeD};
    Object[] inputs = {fourD};
    float[][][][] parsedOutputs = new float[2][8][8][3];
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, parsedOutputs);
    interpreter.runForMultipleInputsOutputs(inputs, outputs);
    float[] outputOneD = parsedOutputs[0][0][0];
    float[] expected = {3.69f, 19.62f, 23.43f};
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
    interpreter.close();
  }

  @Test
  public void testMobilenetRun() {
    // Create a gray image.
    float[][][][] img = new float[1][224][224][3];
    for (int i = 0; i < 224; ++i) {
      for (int j = 0; j < 224; ++j) {
        img[0][i][j][0] = 0.5f;
        img[0][i][j][1] = 0.5f;
        img[0][i][j][2] = 0.5f;
      }
    }

    // Allocate memory to receive the output values.
    float[][] labels = new float[1][1001];

    Interpreter interpreter = new Interpreter(MOBILENET_MODEL_FILE);
    interpreter.run(img, labels);
    interpreter.close();

    assertThat(labels[0])
        .usingExactEquality()
        .containsNoneOf(new float[] {Float.NaN, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY});
  }

  @Test
  public void testRunWithWrongInputType() {
    Interpreter interpreter = new Interpreter(MODEL_FILE);
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
              "DataType (2) of input data does not match with the DataType (1) of model inputs.");
    }
    interpreter.close();
  }

  @Test
  public void testRunWithWrongOutputType() {
    Interpreter interpreter = new Interpreter(MODEL_FILE);
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
              "Cannot convert an TensorFlowLite tensor with type "
                  + "FLOAT32 to a Java object of type [[[[I (which is compatible with the"
                  + " TensorFlowLite type INT32)");
    }
    interpreter.close();
  }

  @Test
  public void testGetInputIndex() {
    Interpreter interpreter = new Interpreter(MOBILENET_MODEL_FILE);
    try {
      interpreter.getInputIndex("WrongInputName");
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "WrongInputName is not a valid name for any input. The indexes of the inputs"
                  + " are {input=0}");
    }
    int index = interpreter.getInputIndex("input");
    assertThat(index).isEqualTo(0);
  }

  @Test
  public void testGetOutputIndex() {
    Interpreter interpreter = new Interpreter(MOBILENET_MODEL_FILE);
    try {
      interpreter.getOutputIndex("WrongOutputName");
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "WrongOutputName is not a valid name for any output. The indexes of the outputs"
                  + " are {MobilenetV1/Predictions/Softmax=0}");
    }
    int index = interpreter.getOutputIndex("MobilenetV1/Predictions/Softmax");
    assertThat(index).isEqualTo(0);
  }

  @Test
  public void testTurnOffNNAPI() throws Exception {
    Path path = MODEL_FILE.toPath();
    FileChannel fileChannel =
        (FileChannel) Files.newByteChannel(path, EnumSet.of(StandardOpenOption.READ));
    MappedByteBuffer mappedByteBuffer =
        fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size());
    Interpreter interpreter = new Interpreter(mappedByteBuffer);
    interpreter.setUseNNAPI(true);
    float[] oneD = {1.23f, 6.54f, 7.81f};
    float[][] twoD = {oneD, oneD, oneD, oneD, oneD, oneD, oneD, oneD};
    float[][][] threeD = {twoD, twoD, twoD, twoD, twoD, twoD, twoD, twoD};
    float[][][][] fourD = {threeD, threeD};
    float[][][][] parsedOutputs = new float[2][8][8][3];
    interpreter.run(fourD, parsedOutputs);
    float[] outputOneD = parsedOutputs[0][0][0];
    float[] expected = {3.69f, 19.62f, 23.43f};
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
    interpreter.setUseNNAPI(false);
    interpreter.run(fourD, parsedOutputs);
    outputOneD = parsedOutputs[0][0][0];
    assertThat(outputOneD).usingTolerance(0.1f).containsExactly(expected).inOrder();
    interpreter.close();
    fileChannel.close();
  }

  @Test
  public void testTurnOnNNAPI() throws Exception {
    Path path = MODEL_FILE.toPath();
    FileChannel fileChannel =
        (FileChannel) Files.newByteChannel(path, EnumSet.of(StandardOpenOption.READ));
    MappedByteBuffer mappedByteBuffer =
        fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size());
    Interpreter interpreter = new Interpreter(mappedByteBuffer);
    interpreter.setUseNNAPI(true);
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
    fileChannel.close();
  }
}
