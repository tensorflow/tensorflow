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

import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;

/**
 * Class that benchmarks image classifier models.
 *
 * <p>===================== General workflow =======================
 *
 * <pre>{@code
 * benchmarker = new OvicBenchmarker();
 * benchmarker.getReadyToTest(labelInputStream, model);
 * while (!benchmarker.shouldStop()) {
 *   Bitmap bitmap = ...
 *   benchmarker.doTestIteration(bitmap);
 * }
 * }</pre>
 */
public class OvicBenchmarker {
  /** Tag for the {@link Log}. */
  private static final String TAG = "OvicBenchmarker";

  /** Evaluation transformation parameters. */
  private static final float CENTRAL_FRACTION = 0.875f;

  /** Dimensions of inputs. */
  private static final int DIM_BATCH_SIZE = 1;
  private static final int DIM_PIXEL_SIZE = 3;
  private int imgHeight = 224;
  private int imgWidth = 224;

  /* Preallocated buffers for storing image data in. */
  private int[] intValues = null;

  /** A ByteBuffer to hold image data, to be feed into classifier as inputs. */
  private ByteBuffer imgData = null;

  private OvicClassifier classifier;

  /** Total runtime in ms. */
  private double totalRuntime = 0.0;
  /** Total allowed runtime in ms. */
  private double wallTime = 20000 * 30.0;

  private Boolean benchmarkStarted = null;

  /**
   * Initializes an {@link OvicBenchmarker}
   *
   * @param wallTime: a double number specifying the total amount of time to benchmark.
   */
  public OvicBenchmarker(double wallTime) {
    benchmarkStarted = false;
    totalRuntime = 0.0;
    this.wallTime = wallTime;
  }

  /** Check whether the benchmarker should stop. */
  public Boolean shouldStop() {
    if (totalRuntime >= wallTime) {
      Log.e(
          TAG,
          "Total runtime "
              + Double.toString(totalRuntime)
              + " exceeded walltime "
              + Double.toString(wallTime));
      return true;
    }
    return false;
  }

  /** Check whether the benchmarker is ready to start classifying images. */
  public Boolean readyToTest() {
    return (classifier != null);
  }

  /**
   * Getting the benchmarker ready for classifying images.
   *
   * @param labelInputStream: an {@link InputStream} specifying where the list of labels should be
   *     read from.
   * @param model: a {@link MappedByteBuffer} model to benchmark.
   */
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

  /** Return how many classes are predicted per image. */
  public int getNumPredictions() {
    return classifier.getNumPredictions();
  }

  /**
   * Perform test on a single bitmap image.
   *
   * @param bitmap: a {@link Bitmap} image to classify.
   */
  public OvicSingleImageResult doTestIteration(Bitmap bitmap)
      throws IOException, InterruptedException {
    if (shouldStop() || !readyToTest()) {
      return null;
    }
    OvicSingleImageResult iterResult = null;
    try {
      Log.i(TAG, "Converting bitmap.");
      convertBitmapToInput(bitmap);
      Log.i(TAG, "Classifying image.");
      iterResult = classifier.classifyByteBuffer(imgData);
    } catch (RuntimeException e) {
      Log.e(TAG, e.getMessage());
      Log.e(TAG, "Failed to classify image.");
    }
    if (iterResult == null || iterResult.latency == null) {
      throw new RuntimeException("Classification result or timing is invalid.");
    }
    Log.d(TAG, "Native inference latency: " + iterResult.latency);
    Log.i(TAG, iterResult.toString());

    if (!benchmarkStarted) {  // Skip the first image to discount warming-up time.
      benchmarkStarted = true;
    } else {
      totalRuntime += (double) iterResult.latency;
    }
    return iterResult;
  }

  /**
   * Writes Image data into a {@link ByteBuffer}.
   *
   * @param bitmap: a {@link Bitmap} source image.
   */
  private void convertBitmapToInput(Bitmap bitmap) throws RuntimeException {
    if (imgData == null) {
      throw new RuntimeException("Benchmarker is not yet ready to test.");
    }
    imgData.rewind();
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

    // Convert the image to ByteBuffer.
    int pixel = 0;
    long startTime = SystemClock.uptimeMillis();

    for (int i = 0; i < imgHeight; ++i) {
      for (int j = 0; j < imgWidth; ++j) {
        final int val = intValues[pixel++];
        imgData.put((byte) ((val >> 16) & 0xFF));
        imgData.put((byte) ((val >> 8) & 0xFF));
        imgData.put((byte) (val & 0xFF));
      }
    }
    long endTime = SystemClock.uptimeMillis();
    Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
  }
}
