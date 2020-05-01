/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package org.tensorflow.ovic;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import javax.imageio.ImageIO;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.ovic.OvicClassifier}. */
@RunWith(JUnit4.class)
public final class OvicClassifierTest {

  private OvicClassifier classifier;
  private InputStream labelsInputStream = null;
  private MappedByteBuffer quantizedModel = null;
  private MappedByteBuffer floatModel = null;
  private MappedByteBuffer lowResModel = null;
  private ByteBuffer testImage = null;
  private ByteBuffer lowResTestImage = null;
  private OvicClassificationResult testResult = null;
  private static final String LABELS_PATH =
      "tensorflow/lite/java/ovic/src/testdata/labels.txt";
  private static final String QUANTIZED_MODEL_PATH =
      "external/tflite_ovic_testdata/quantized_model.lite";
  private static final String LOW_RES_MODEL_PATH =
      "external/tflite_ovic_testdata/low_res_model.lite";
  private static final String FLOAT_MODEL_PATH =
      "external/tflite_ovic_testdata/float_model.lite";
  private static final String TEST_IMAGE_PATH =
      "external/tflite_ovic_testdata/test_image_224.jpg";
  private static final String TEST_LOW_RES_IMAGE_PATH =
      "external/tflite_ovic_testdata/test_image_128.jpg";
  private static final int TEST_IMAGE_GROUNDTRUTH = 653; // "military uniform"

  @Before
  public void setUp() {
    try {
      File labelsfile = new File(LABELS_PATH);
      labelsInputStream = new FileInputStream(labelsfile);
      quantizedModel = loadModelFile(QUANTIZED_MODEL_PATH);
      floatModel = loadModelFile(FLOAT_MODEL_PATH);
      lowResModel = loadModelFile(LOW_RES_MODEL_PATH);
      File imageFile = new File(TEST_IMAGE_PATH);
      BufferedImage img = ImageIO.read(imageFile);
      testImage = toByteBuffer(img);
      // Low res image and models.
      imageFile = new File(TEST_LOW_RES_IMAGE_PATH);
      img = ImageIO.read(imageFile);
      lowResTestImage = toByteBuffer(img);
    } catch (IOException e) {
      System.out.print(e.getMessage());
    }
    System.out.println("Successful setup");
  }

  @Test
  public void ovicClassifier_quantizedModelCreateSuccess() throws Exception {
    classifier = new OvicClassifier(labelsInputStream, quantizedModel);
    assertThat(classifier).isNotNull();
  }

  @Test
  public void ovicClassifier_floatModelCreateSuccess() throws Exception {
    classifier = new OvicClassifier(labelsInputStream, floatModel);
    assertThat(classifier).isNotNull();
  }

  @Test
  public void ovicClassifier_quantizedModelClassifySuccess() throws Exception {
    classifier = new OvicClassifier(labelsInputStream, quantizedModel);
    testResult = classifier.classifyByteBuffer(testImage);
    assertCorrectTopK(testResult);
  }

  @Test
  public void ovicClassifier_floatModelClassifySuccess() throws Exception {
    classifier = new OvicClassifier(labelsInputStream, floatModel);
    testResult = classifier.classifyByteBuffer(testImage);
    assertCorrectTopK(testResult);
  }

  @Test
  public void ovicClassifier_lowResModelClassifySuccess() throws Exception {
    classifier = new OvicClassifier(labelsInputStream, lowResModel);
    testResult = classifier.classifyByteBuffer(lowResTestImage);
    assertCorrectTopK(testResult);
  }

  @Test
  public void ovicClassifier_latencyNotNull() throws Exception {
    classifier = new OvicClassifier(labelsInputStream, floatModel);
    testResult = classifier.classifyByteBuffer(testImage);
    assertThat(testResult.latencyNano).isNotNull();
  }

  @Test
  public void ovicClassifier_mismatchedInputResolutionFails() throws Exception {
    classifier = new OvicClassifier(labelsInputStream, lowResModel);
    int[] inputDims = classifier.getInputDims();
    assertThat(inputDims[1]).isEqualTo(128);
    assertThat(inputDims[2]).isEqualTo(128);
    try {
      testResult = classifier.classifyByteBuffer(testImage);
      fail();
    } catch (IllegalArgumentException e) {
      // Success.
    }
  }

  private static ByteBuffer toByteBuffer(BufferedImage image) {
    ByteBuffer imgData = ByteBuffer.allocateDirect(
        image.getHeight() * image.getWidth() * 3);
    imgData.order(ByteOrder.nativeOrder());
    for (int y = 0; y < image.getHeight(); y++) {
      for (int x = 0; x < image.getWidth(); x++) {
        int val = image.getRGB(x, y);
        imgData.put((byte) ((val >> 16) & 0xFF));
        imgData.put((byte) ((val >> 8) & 0xFF));
        imgData.put((byte) (val & 0xFF));
      }
    }
    return imgData;
  }

  private static void assertCorrectTopK(OvicClassificationResult testResult) {
    assertThat(testResult.topKClasses.size()).isGreaterThan(0);
    Boolean topKAccurate = false;
    // Assert that the correct class is in the top K.
    for (int i = 0; i < testResult.topKIndices.size(); i++) {
      if (testResult.topKIndices.get(i) == TEST_IMAGE_GROUNDTRUTH) {
        topKAccurate = true;
        break;
      }
    }
    System.out.println(testResult.toString());
    System.out.flush();
    assertThat(topKAccurate).isTrue();
  }

  private static MappedByteBuffer loadModelFile(String modelFilePath) throws IOException {
    File modelfile = new File(modelFilePath);
    FileInputStream inputStream = new FileInputStream(modelfile);
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = 0L;
    long declaredLength = fileChannel.size();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }
}
