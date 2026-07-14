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

import android.graphics.Bitmap;
import android.util.Log;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;

/** Class that benchmarks image classifier models. */
public final class OvicClassifierBenchmarker extends OvicBenchmarker {
  /** Tag for the {@link Log}. */
  private static final String TAG = "OvicClassifierBenchmarker";

  /** ImageNet preprocessing parameters. */
  private static final float CENTRAL_FRACTION = 0.875f;
  private OvicClassifier classifier;
  private OvicClassificationResult iterResult = null;

  public OvicClassifierBenchmarker(double wallTime) {
    super(wallTime);
  }

  /** Test if the classifier is ready for benchmarking. */
  @Override
  public boolean readyToTest() {
    return (classifier != null);
  }

  /**
   * Getting the benchmarker ready for classifying images.
   *
   * @param labelInputStream: an {@link InputStream} specifying where the list of labels should be
   *     read from.
   * @param model: a {@link MappedByteBuffer} model to benchmark.
   */
  @Override
   public void getReadyToTest(InputStream labelInputStream, MappedByteBuffer model) {
    try {
      Log.i(TAG, "Creating classifier.");
      classifier = new OvicClassifier(labelInputStream, model);
      int [] inputDims = classifier.getInputDims();
      imgHeight = inputDims[1];
      imgWidth = inputDims[2];
      // Only accept QUANTIZED_UINT8 input.
      imgData = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * imgHeight * imgWidth * DIM_PIXEL_SIZE);
      imgData.order(ByteOrder.nativeOrder());
      intValues = new int[imgHeight * imgWidth];
    } catch (Exception e) {
        Log.e(TAG, e.getMessage());
        Log.e(TAG, "Failed to initialize ImageNet classifier for the benchmarker.");
    }
  }

  /**
   * Perform classification on a single bitmap image.
   *
   * @param bitmap: a {@link Bitmap} image to process.
   * @param imageId: an ID uniquely representing the image.
   */
  @Override
  public boolean processBitmap(Bitmap bitmap, int imageId)
      throws IOException, InterruptedException {
    if (shouldStop() || !readyToTest()) {
      return false;
    }
    try {
      Log.i(TAG, "Converting bitmap.");
      convertBitmapToInput(bitmap);
      Log.i(TAG, "Classifying image: " + imageId);
      iterResult = classifier.classifyByteBuffer(imgData);
    } catch (RuntimeException e) {
      Log.e(TAG, e.getMessage());
      Log.e(TAG, "Failed to classify image.");
    }
    if (iterResult == null || iterResult.latencyMilli == null || iterResult.latencyNano == null) {
      throw new RuntimeException("Classification result or timing is invalid.");
    }
    Log.d(TAG, "Native inference latency (ms): " + iterResult.latencyMilli);
    Log.d(TAG, "Native inference latency (ns): " + iterResult.latencyNano);
    Log.i(TAG, iterResult.toString());

    if (!benchmarkStarted) {  // Skip the first image to discount warming-up time.
      benchmarkStarted = true;
    } else {
      totalRuntimeNano += ((double) iterResult.latencyNano);
    }
    return true;
  }

  /** Return how many classes are predicted per image. */
  public int getNumPredictions() {
    return classifier.getNumPredictions();
  }

  public OvicClassificationResult getLastClassificationResult() {
    return iterResult;
  }

  @Override
  public String getLastResultString() {
    if (iterResult == null) {
      return null;
    } else {
      return iterResult.toString();
    }
  }

  /**
   * Preprocess bitmap according to ImageNet protocol then writes result into a {@link ByteBuffer}.
   *
   * @param bitmap: a {@link Bitmap} source image.
   */
  private void convertBitmapToInput(Bitmap bitmap) {
    // Perform transformations corresponding to evaluation mode.
    float width = (float) bitmap.getWidth();
    float height = (float) bitmap.getHeight();
    int stWidth = Math.round((width - width * CENTRAL_FRACTION) / 2);
    int stHeight = Math.round((height - height * CENTRAL_FRACTION) / 2);
    int newWidth = Math.round(width - stWidth * 2);
    int newHeight = Math.round(height - stHeight * 2);
    bitmap = Bitmap.createBitmap(bitmap, stWidth, stHeight, newWidth, newHeight);
    bitmap = Bitmap.createScaledBitmap(bitmap, imgWidth, imgHeight, true);
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    loadsInputToByteBuffer();
  }
}
