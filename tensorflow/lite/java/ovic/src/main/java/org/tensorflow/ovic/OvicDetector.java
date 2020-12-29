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

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.TestHelper;

/** Class for running COCO detection with a TfLite model. */
public class OvicDetector implements AutoCloseable {

  /** Tag for the {@link Log}. */
  private static final String TAG = "OvicDetector";

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  private Interpreter tflite;

  /** Labels corresponding to the output of the vision model. */
  private final List<String> labelList;

  /** Number of detections per image. 10 for demo, 100 for the actual competition. */
  private static final int NUM_RESULTS = 100;

  /** The output arrays for the mobilenet SSD. */
  private float[][][] outputLocations;
  private float[][] outputClasses;
  private float[][] outputScores;
  private float[] numDetections;
  private Map<Integer, Object> outputMap;

  /** Input resolution. */
  private final int[] inputDims;

  /** Final result. */
  public OvicDetectionResult result = null;

  OvicDetector(InputStream labelInputStream, MappedByteBuffer model) throws IOException {
    // Load the label list.
    labelList = loadLabelList(labelInputStream);

    // Create the TfLite interpreter.
    tflite = new Interpreter(model, new Interpreter.Options().setNumThreads(1));
    inputDims = TestHelper.getInputDims(tflite, 0);
    if (TestHelper.getInputDataType(tflite, 0).equals("float")) {
      throw new RuntimeException("The model's input must be QUANTIZED_UINT8.");
    }
    if (inputDims.length != 4) {
      throw new RuntimeException("The model's input dimensions must be 4 (BWHC).");
    }
    if (inputDims[0] != 1) {
      throw new RuntimeException(
          "The model must have a batch size of 1, got " + inputDims[0] + " instead.");
    }
    if (inputDims[3] != 3) {
      throw new RuntimeException(
          "The model must have three color channels, got " + inputDims[3] + " instead.");
    }
    // Check the resolution.
    int minSide = Math.min(inputDims[1], inputDims[2]);
    int maxSide = Math.max(inputDims[1], inputDims[2]);
    if (minSide <= 0 || maxSide > 1000) {
      throw new RuntimeException("The model's resolution must be between (0, 1000].");
    }

    // Initialize the input array and result arrays. The input images are stored in a list of
    // Object. Since this function anaylzed one image per time, there is only 1 item.
    // The output is fomulated as a map of int -> Object. The output arrays are added to the map.
    outputLocations = new float[1][NUM_RESULTS][4];
    outputClasses = new float[1][NUM_RESULTS];
    outputScores = new float[1][NUM_RESULTS];
    numDetections = new float[1];
    outputMap = new HashMap<>();
    outputMap.put(0, outputLocations);
    outputMap.put(1, outputClasses);
    outputMap.put(2, outputScores);
    outputMap.put(3, numDetections);
    // Preallocate the result. This will be where inference result is stored after each
    // detectByteBuffer call.
    result = new OvicDetectionResult(NUM_RESULTS);
  }

  /** Reads label list from Assets. */
  private static List<String> loadLabelList(InputStream labelInputStream) throws IOException {
    List<String> labelList = new ArrayList<>();
    try (BufferedReader reader =
        new BufferedReader(new InputStreamReader(labelInputStream, UTF_8))) {
      String line;
      while ((line = reader.readLine()) != null) {
        labelList.add(line);
      }
    }
    return labelList;
  }

  /**
   * The interface to run the detection. This method currently only support float mobilenet_ssd
   * model. The quantized models will be added in the future.
   *
   * @param imgData The image buffer in ByteBuffer format.
   * @return boolean indicator of whether detection was a success. If success, the detection results
   *  is available in the result member variable.
   *     See OvicDetectionResult.java for details.
   */
  boolean detectByteBuffer(ByteBuffer imgData, int imageId) {
    if (tflite == null) {
      throw new RuntimeException(TAG + ": Detector has not been initialized; Failed.");
    }

    Object[] inputArray = {imgData};
    tflite.runForMultipleInputsOutputs(inputArray, outputMap);

    Long latencyMilli = getLastNativeInferenceLatencyMilliseconds();
    Long latencyNano = getLastNativeInferenceLatencyNanoseconds();

    // Update the results.
    result.resetTo(latencyMilli, latencyNano, imageId);
    for (int i = 0; i < NUM_RESULTS; i++) {
      // The model returns normalized coordinates [start_y, start_x, end_y, end_x].
      // The boxes expect pixel coordinates [x1, y1, x2, y2].
      // The height and width of the input are in inputDims[1] and inputDims[2].
      // The following command converts between model outputs to bounding boxes.
      result.addBox(
          outputLocations[0][i][1] * inputDims[2],
          outputLocations[0][i][0] * inputDims[1],
          outputLocations[0][i][3] * inputDims[2],
          outputLocations[0][i][2] * inputDims[1],
          Math.round(outputClasses[0][i] + 1 /* Label offset */),
          outputScores[0][i]);
    }
    return true;  // Marks that the result is available.
  }

  /*
   * Get native inference latency of last image detection run.
   *  @throws RuntimeException if model is uninitialized.
   *  @return The inference latency in milliseconds.
   */
  public Long getLastNativeInferenceLatencyMilliseconds() {
    if (tflite == null) {
      throw new RuntimeException(TAG + ": ImageNet classifier has not been initialized; Failed.");
    }
    Long latency = tflite.getLastNativeInferenceDurationNanoseconds();
    return (latency == null) ? null : (Long) (latency / 1000000);
  }

  /*
   * Get native inference latency of last image detection run.
   *  @throws RuntimeException if model is uninitialized.
   *  @return The inference latency in nanoseconds.
   */
  public Long getLastNativeInferenceLatencyNanoseconds() {
    if (tflite == null) {
      throw new IllegalStateException(
          TAG + ": ImageNet classifier has not been initialized; Failed.");
    }
    return tflite.getLastNativeInferenceDurationNanoseconds();
  }

  public int[] getInputDims() {
    return inputDims;
  }

  public List<String> getLabels() {
    return labelList;
  }

  /** Closes tflite to release resources. */
  @Override
  public void close() {
    tflite.close();
    tflite = null;
  }
}
