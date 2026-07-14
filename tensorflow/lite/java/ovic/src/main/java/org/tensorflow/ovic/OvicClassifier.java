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

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.TestHelper;

/** Class for running ImageNet classification with a TfLite model. */
public class OvicClassifier {

  /** Tag for the {@link Log}. */
  private static final String TAG = "OvicClassifier";

  /** Number of results to show (i.e. the "K" in top-K predictions). */
  private static final int RESULTS_TO_SHOW = 5;

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  private Interpreter tflite;

  /** Labels corresponding to the output of the vision model. */
  private final List<String> labelList;

  /** An array to hold inference results, to be feed into Tensorflow Lite as outputs. */
  private byte[][] inferenceOutputArray = null;
  /** An array to hold final prediction probabilities. */
  private float[][] labelProbArray = null;

  /** Input resultion. */
  private int[] inputDims = null;
  /** Whether the model runs as float or quantized. */
  private Boolean outputIsFloat = null;

  private final PriorityQueue<Map.Entry<Integer, Float>> sortedLabels =
      new PriorityQueue<>(
          RESULTS_TO_SHOW,
          new Comparator<Map.Entry<Integer, Float>>() {
            @Override
            public int compare(Map.Entry<Integer, Float> o1, Map.Entry<Integer, Float> o2) {
              return o1.getValue().compareTo(o2.getValue());
            }
          });

  /** Initializes an {@code OvicClassifier}. */
  public OvicClassifier(InputStream labelInputStream, MappedByteBuffer model) throws IOException {
    if (model == null) {
      throw new RuntimeException("Input model is empty.");
    }
    labelList = loadLabelList(labelInputStream);
    // OVIC uses one thread for CPU inference.
    tflite = new Interpreter(model, new Interpreter.Options().setNumThreads(1));
    inputDims = TestHelper.getInputDims(tflite, 0);
    if (inputDims.length != 4) {
      throw new RuntimeException("The model's input dimensions must be 4 (BWHC).");
    }
    if (inputDims[0] != 1) {
      throw new IllegalStateException(
          "The model must have a batch size of 1, got " + inputDims[0] + " instead.");
    }
    if (inputDims[3] != 3) {
      throw new IllegalStateException(
          "The model must have three color channels, got " + inputDims[3] + " instead.");
    }
    int minSide = Math.min(inputDims[1], inputDims[2]);
    int maxSide = Math.max(inputDims[1], inputDims[2]);
    if (minSide <= 0 || maxSide > 1000) {
      throw new RuntimeException("The model's resolution must be between (0, 1000].");
    }
    String outputDataType = TestHelper.getOutputDataType(tflite, 0);
    switch (outputDataType) {
      case "float":
        outputIsFloat = true;
        break;
      case "byte":
        outputIsFloat = false;
        break;
      default:
        throw new IllegalStateException("Cannot process output type: " + outputDataType);
    }
    inferenceOutputArray = new byte[1][labelList.size()];
    labelProbArray = new float[1][labelList.size()];
  }

  /** Classifies a {@link ByteBuffer} image. */
  // @throws RuntimeException if model is uninitialized.
  public OvicClassificationResult classifyByteBuffer(ByteBuffer imgData) {
    if (tflite == null) {
      throw new RuntimeException(TAG + ": ImageNet classifier has not been initialized; Failed.");
    }
    if (outputIsFloat == null) {
      throw new RuntimeException(TAG + ": Classifier output type has not been resolved.");
    }
    if (outputIsFloat) {
      tflite.run(imgData, labelProbArray);
    } else {
      tflite.run(imgData, inferenceOutputArray);
      /** Convert results to float */
      for (int i = 0; i < inferenceOutputArray[0].length; i++) {
        labelProbArray[0][i] = (inferenceOutputArray[0][i] & 0xff) / 255.0f;
      }
    }
    OvicClassificationResult iterResult = computeTopKLabels();
    iterResult.latencyMilli = getLastNativeInferenceLatencyMilliseconds();
    iterResult.latencyNano = getLastNativeInferenceLatencyNanoseconds();
    return iterResult;
  }

  /** Return the probability array of all classes. */
  public float[][] getlabelProbArray() {
    return labelProbArray;
  }

  /** Return the number of top labels predicted by the classifier. */
  public int getNumPredictions() {
    return RESULTS_TO_SHOW;
  }

  /** Return the four dimensions of the input image. */
  public int[] getInputDims() {
    return inputDims;
  }

  /**
   * Get native inference latency of last image classification run.
   *
   * @throws RuntimeException if model is uninitialized.
   */
  public Long getLastNativeInferenceLatencyMilliseconds() {
    if (tflite == null) {
      throw new RuntimeException(TAG + ": ImageNet classifier has not been initialized; Failed.");
    }
    Long latency = tflite.getLastNativeInferenceDurationNanoseconds();
    return (latency == null) ? null : (Long) (latency / 1000000);
  }

  /**
   * Get native inference latency of last image classification run.
   *
   * @throws RuntimeException if model is uninitialized.
   */
  public Long getLastNativeInferenceLatencyNanoseconds() {
    if (tflite == null) {
      throw new IllegalStateException(
          TAG + ": ImageNet classifier has not been initialized; Failed.");
    }
    return tflite.getLastNativeInferenceDurationNanoseconds();
  }

  /** Closes tflite to release resources. */
  public void close() {
    tflite.close();
    tflite = null;
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

  /** Computes top-K labels. */
  private OvicClassificationResult computeTopKLabels() {
    if (labelList == null) {
      throw new RuntimeException("Label file has not been loaded.");
    }
    for (int i = 0; i < labelList.size(); ++i) {
      sortedLabels.add(new AbstractMap.SimpleEntry<>(i, labelProbArray[0][i]));
      if (sortedLabels.size() > RESULTS_TO_SHOW) {
        sortedLabels.poll();
      }
    }
    OvicClassificationResult singleImageResult = new OvicClassificationResult();
    if (sortedLabels.size() != RESULTS_TO_SHOW) {
      throw new RuntimeException(
          "Number of returned labels does not match requirement: "
              + sortedLabels.size()
              + " returned, but "
              + RESULTS_TO_SHOW
              + " required.");
    }
    for (int i = 0; i < RESULTS_TO_SHOW; ++i) {
      Map.Entry<Integer, Float> label = sortedLabels.poll();
      // ImageNet model prediction indices are 0-based.
      singleImageResult.topKIndices.add(label.getKey());
      singleImageResult.topKClasses.add(labelList.get(label.getKey()));
      singleImageResult.topKProbs.add(label.getValue());
    }
    // Labels with lowest probability are returned first, hence need to reverse them.
    Collections.reverse(singleImageResult.topKIndices);
    Collections.reverse(singleImageResult.topKClasses);
    Collections.reverse(singleImageResult.topKProbs);
    return singleImageResult;
  }
}
