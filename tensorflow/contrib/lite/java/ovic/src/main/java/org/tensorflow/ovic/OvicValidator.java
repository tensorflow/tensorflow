/*Copyright 2018 Google LLC

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

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Random;

/** Validate a submission model. */
public class OvicValidator {
  private static void printUsage(PrintStream s) {
    s.println("Java program that validates a submission model.");
    s.println();
    s.println("Usage: ovic_validator <submission file>");
    s.println();
    s.println("Where:");
    s.println("<submission file> is the model in TfLite format;");
  }

  public static void main(String[] args) {
    if (args.length != 1) {
      printUsage(System.err);
      System.exit(1);
    }
    final String labelPath =
        "tensorflow/contrib/lite/java/ovic/src/testdata/labels.txt";

    final String modelFile = args[0];
    try {
      File labelsfile = new File(labelPath);
      InputStream labelsInputStream = new FileInputStream(labelsfile);
      MappedByteBuffer model = loadModelFile(modelFile);
      OvicClassifier classifier = new OvicClassifier(labelsInputStream, model);
      ByteBuffer imgData = createByteBufferForClassifier(classifier);
      OvicSingleImageResult testResult = classifier.classifyByteBuffer(imgData);
      if (testResult.topKClasses.isEmpty()) {
        throw new RuntimeException("Failed to return top K predictions.");
      }
      System.out.printf("Successfully validated %s.%n", modelFile);
    } catch (Exception e) {
      System.out.println(e.getMessage());
      System.out.printf("Failed to validate %s.%n", modelFile);
    }
  }

  private static ByteBuffer createByteBufferForClassifier(OvicClassifier classifier) {
    if (classifier == null) {
      throw new RuntimeException("Cannot create image buffer with the classifier.");
    }
    int[] inputDims = classifier.getInputDims();
    int imgHeight = inputDims[1];
    int imgWidth = inputDims[2];
    ByteBuffer imgData = ByteBuffer.allocateDirect(imgHeight * imgWidth * 3);
    imgData.order(ByteOrder.nativeOrder());
    Random rand = new Random();
    for (int y = 0; y < imgHeight; y++) {
      for (int x = 0; x < imgWidth; x++) {
        int val = rand.nextInt();
        imgData.put((byte) ((val >> 16) & 0xFF));
        imgData.put((byte) ((val >> 8) & 0xFF));
        imgData.put((byte) (val & 0xFF));
      }
    }
    return imgData;
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
