/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.demo.env.Logger;

/**
 * A detector for general purpose object detection as described in Scalable Object Detection using
 * Deep Neural Networks (https://arxiv.org/abs/1312.2249).
 */
public class TensorFlowMultiBoxDetector implements Classifier {
  private static final Logger LOGGER = new Logger();

  static {
    System.loadLibrary("tensorflow_demo");
  }

  // Only return this many results with at least this confidence.
  private static final int MAX_RESULTS = Integer.MAX_VALUE;

  // Config values.
  private String inputName;
  private int inputSize;
  private int imageMean;
  private float imageStd;

  // Pre-allocated buffers.
  private int[] intValues;
  private float[] floatValues;
  private float[] outputLocations;
  private float[] outputScores;
  private String[] outputNames;
  private int numLocations;

  private TensorFlowInferenceInterface inferenceInterface;

  private float[] boxPriors;

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param locationFilename The filepath of label file for classes.
   * @param inputSize The input size. A square image of inputSize x inputSize is assumed.
   * @param imageMean The assumed mean of the image values.
   * @param imageStd The assumed std of the image values.
   * @param inputName The label of the image input node.
   * @param outputName The label of the output node.
   * @return The native return value, 0 indicating success.
   * @throws IOException
   */
  public int initializeTensorFlow(
      final AssetManager assetManager,
      final String modelFilename,
      final String locationFilename,
      final int numLocations,
      final int inputSize,
      final int imageMean,
      final float imageStd,
      final String inputName,
      final String outputName)
      throws IOException {
    this.inputName = inputName;
    this.inputSize = inputSize;
    this.imageMean = imageMean;
    this.imageStd = imageStd;
    this.numLocations = numLocations;

    this.boxPriors = new float[numLocations * 8];

    loadCoderOptions(assetManager, locationFilename, boxPriors);

    // Pre-allocate buffers.
    outputNames = outputName.split(",");
    intValues = new int[inputSize * inputSize];
    floatValues = new float[inputSize * inputSize * 3];
    outputScores = new float[numLocations];
    outputLocations = new float[numLocations * 4];

    inferenceInterface = new TensorFlowInferenceInterface();

    return inferenceInterface.initializeTensorFlow(assetManager, modelFilename);
  }

  // Load BoxCoderOptions from native code.
  private native void loadCoderOptions(
      AssetManager assetManager, String locationFilename, float[] boxPriors);

  private float[] decodeLocationsEncoding(final float[] locationEncoding) {
    final float[] locations = new float[locationEncoding.length];
    boolean nonZero = false;
    for (int i = 0; i < numLocations; ++i) {
      for (int j = 0; j < 4; ++j) {
        final float currEncoding = locationEncoding[4 * i + j];
        nonZero = nonZero || currEncoding != 0.0f;

        final float mean = boxPriors[i * 8 + j * 2];
        final float stdDev = boxPriors[i * 8 + j * 2 + 1];
        float currentLocation = currEncoding * stdDev + mean;
        currentLocation = Math.max(currentLocation, 0.0f);
        currentLocation = Math.min(currentLocation, 1.0f);
        locations[4 * i + j] = currentLocation;
      }
    }

    if (!nonZero) {
      LOGGER.w("No non-zero encodings; check log for inference errors.");
    }
    return locations;
  }

  private float[] decodeScoresEncoding(final float[] scoresEncoding) {
    final float[] scores = new float[scoresEncoding.length];
    for (int i = 0; i < scoresEncoding.length; ++i) {
      scores[i] = 1 / ((float) (1 + Math.exp(-scoresEncoding[i])));
    }
    return scores;
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    for (int i = 0; i < intValues.length; ++i) {
      floatValues[i * 3 + 0] = ((intValues[i] & 0xFF) - imageMean) / imageStd;
      floatValues[i * 3 + 1] = (((intValues[i] >> 8) & 0xFF) - imageMean) / imageStd;
      floatValues[i * 3 + 2] = (((intValues[i] >> 16) & 0xFF) - imageMean) / imageStd;
    }
    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("fillNodeFloat");
    inferenceInterface.fillNodeFloat(
        inputName, new int[] {1, inputSize, inputSize, 3}, floatValues);
    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("runInference");
    inferenceInterface.runInference(outputNames);
    Trace.endSection();

    // Copy the output Tensor back into the output array.
    Trace.beginSection("readNodeFloat");
    final float[] outputScoresEncoding = new float[numLocations];
    final float[] outputLocationsEncoding = new float[numLocations * 4];
    inferenceInterface.readNodeFloat(outputNames[0], outputLocationsEncoding);
    inferenceInterface.readNodeFloat(outputNames[1], outputScoresEncoding);
    Trace.endSection();

    outputLocations = decodeLocationsEncoding(outputLocationsEncoding);
    outputScores = decodeScoresEncoding(outputScoresEncoding);

    // Find the best detections.
    final PriorityQueue<Recognition> pq =
        new PriorityQueue<Recognition>(
            1,
            new Comparator<Recognition>() {
              @Override
              public int compare(final Recognition lhs, final Recognition rhs) {
                // Intentionally reversed to put high confidence at the head of the queue.
                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
              }
            });

    // Scale them back to the input size.
    for (int i = 0; i < outputScores.length; ++i) {
      final RectF detection =
          new RectF(
              outputLocations[4 * i] * inputSize,
              outputLocations[4 * i + 1] * inputSize,
              outputLocations[4 * i + 2] * inputSize,
              outputLocations[4 * i + 3] * inputSize);
      pq.add(new Recognition("" + i, "" + i, outputScores[i], detection));
    }

    final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
    for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
      recognitions.add(pq.poll());
    }
    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

  @Override
  public void close() {
    inferenceInterface.close();
  }
}
