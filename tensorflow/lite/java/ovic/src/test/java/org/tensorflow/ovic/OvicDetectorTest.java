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

import java.awt.Graphics2D;
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

/** Unit test for {@link org.tensorflow.ovic.OvicDetector}. */
@RunWith(JUnit4.class)
public final class OvicDetectorTest {
  private OvicDetector detector = null;
  private InputStream labelsInputStream = null;
  private MappedByteBuffer model = null;
  private ByteBuffer testImage = null;

  private static final String LABELS_PATH =
      "tensorflow/lite/java/ovic/src/testdata/coco_labels.txt";
  private static final String MODEL_PATH =
      "external/tflite_ovic_testdata/quantized_detect.lite";
  private static final String TEST_IMAGE_PATH =
      "external/tflite_ovic_testdata/test_image_224.jpg";
  private static final int GROUNDTRUTH = 1 /* Person */;

  @Before
  public void setUp() {
    try {
      // load models.
      model = loadModelFile(MODEL_PATH);

      // Load label files;
      File labelsfile = new File(LABELS_PATH);
      labelsInputStream = new FileInputStream(labelsfile);

      // Create detector.
      detector = new OvicDetector(labelsInputStream, model);

      // Load test image and convert into byte buffer.
      File imageFile = new File(TEST_IMAGE_PATH);
      BufferedImage rawimg = ImageIO.read(imageFile);
      int[] inputDims = detector.getInputDims();
      BufferedImage img = new BufferedImage(inputDims[1], inputDims[2], rawimg.getType());
      Graphics2D g = img.createGraphics();
      g.drawImage(rawimg, 0, 0, inputDims[1], inputDims[2], null);
      g.dispose();
      testImage = toByteBuffer(img);
    } catch (IOException e) {
      System.out.println(e.getMessage());
    }

    System.out.println("Successfully setup");
  }

  private static MappedByteBuffer loadModelFile(String modelFilePath) throws IOException {
    File modelfile = new File(modelFilePath);
    FileInputStream inputStream = new FileInputStream(modelfile);
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = 0L;
    long declaredLength = fileChannel.size();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  private static ByteBuffer toByteBuffer(BufferedImage image) {
    ByteBuffer imgData = ByteBuffer.allocateDirect(image.getHeight() * image.getWidth() * 3);
    imgData.order(ByteOrder.nativeOrder());
    for (int y = 0; y < image.getHeight(); y++) {
      for (int x = 0; x < image.getWidth(); x++) {
        int pixelValue = image.getRGB(x, y);
        imgData.put((byte) ((pixelValue >> 16) & 0xFF));
        imgData.put((byte) ((pixelValue >> 8) & 0xFF));
        imgData.put((byte) (pixelValue & 0xFF));
      }
    }
    return imgData;
  }

  @Test
  public void ovicDetector_detectSuccess() throws Exception {
    assertThat(detector.detectByteBuffer(testImage, 1)).isTrue();
    assertThat(detector.result != null).isTrue();
  }

  @Test
  public void ovicDetector_simpleBatchTest() throws Exception {
    final int numRepeats = 5;
    for (int i = 0; i < numRepeats; i++) {
      assertThat(detector.detectByteBuffer(testImage, 1)).isTrue();
      OvicDetectionResult result = detector.result;
      Boolean detectWithinTop5 = false;
      for (int j = 0; j < Math.min(5, result.count); j++) {
        if (result.detections.get(j).category == GROUNDTRUTH) {
          detectWithinTop5 = true;
          break;
        }
      }
      if (!detectWithinTop5) {
        System.out.println("---------------- Image " + i + " ---------------------");
        System.out.println("Expect category " + GROUNDTRUTH);
        System.out.println("Detection results: ");
        System.out.println(result.toString());
      }
      assertThat(detectWithinTop5).isTrue();
    }
  }
}
