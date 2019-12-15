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

/**
 * Class that benchmarks object detection models.
 */
public final class OvicDetectorBenchmarker extends OvicBenchmarker {
  /** Tag for the {@link Log}. */
  private static final String TAG = "OvicDetectorBenchmarker";

  public double scaleFactorWidth = 1.0f;
  public double scaleFactorHeight = 1.0f;
  private Bitmap scaledBitmap = null;  // Preallocate bitmap for scaling.

  private OvicDetector detector;

  /**
   * Initializes an {@link OvicDetectionBenchmarker}
   *
   * @param wallTime: a double number specifying the total amount of time to benchmark.
   */
  public OvicDetectorBenchmarker(double wallTime) {
    super(wallTime);
  }

  /** Check to see if the detector is ready to test. */
  @Override
  public boolean readyToTest() {
    return (detector != null);
  }

  /**
   * Getting the benchmarker ready for detecting images.
   *
   * @param labelInputStream: an {@link InputStream} specifying where the list of labels should be
   *     read from.
   * @param model: a {@link MappedByteBuffer} model to benchmark.
   */
  @Override
  public void getReadyToTest(InputStream labelInputStream, MappedByteBuffer model) {
    try {
      Log.i(TAG, "Creating detector.");
      detector = new OvicDetector(labelInputStream, model);
      int[] inputDims = detector.getInputDims();
      imgHeight = inputDims[1];
      imgWidth = inputDims[2];
      imgData = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * imgHeight * imgWidth * DIM_PIXEL_SIZE);
      imgData.order(ByteOrder.nativeOrder());
      intValues = new int[imgHeight * imgWidth];
      benchmarkStarted = false;
    } catch (Exception e) {
      Log.e(TAG, e.getMessage());
      Log.e(TAG, "Failed to initialize COCO detector for the benchmarker.", e);
    }
  }

  /**
   * Perform detection on a single ByteBuffer {@link ByteBuffer} image. The image must have the
   * same dimension that the model expects.
   *
   * @param image: a {@link ByteBuffer} image to process.
   * @param imageId: an ID uniquely representing the image.
   */
  public boolean processBuffer(ByteBuffer image, int imageId) {
    if (!readyToTest()) {
      return false;
    }
    try {
      if (!detector.detectByteBuffer(image, imageId)) {
        return false;
      }
    } catch (RuntimeException e) {
      Log.e(TAG, e.getMessage());
      return false;
    }

    if (!benchmarkStarted) { // Skip the first image to discount warming-up time.
      benchmarkStarted = true;
    } else {
      totalRuntime += ((double) detector.result.latency);
    }
    return true;  // Indicating that result is ready.
  }

  /**
   * Perform detection on a single bitmap image.
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
    convertBitmapToInput(bitmap);  // Scale bitmap if needed, store result in imgData.
    if (!processBuffer(imgData, imageId)) {
      return false;
    }
    // Scale results back to original image coordinates.
    detector.result.scaleUp(scaleFactorWidth, scaleFactorHeight);
    return true;  // Indicating that result is ready.
  }

  public OvicDetectionResult getLastDetectionResult() {
    return detector.result;
  }

  @Override
  public String getLastResultString() {
    if (detector.result == null) {
      return null;
    }
    return detector.result.toString();
  }

  /**
   * Preprocess bitmap image into {@link ByteBuffer} format for the detector.
   *
   * @param bitmap: a {@link Bitmap} source image.
   */
  private void convertBitmapToInput(Bitmap bitmap) {
    int originalWidth = bitmap.getWidth();
    int originalHeight = bitmap.getHeight();
    scaledBitmap = Bitmap.createScaledBitmap(bitmap, imgWidth, imgHeight, true);
    scaleFactorWidth = originalWidth * 1.0 / imgWidth;
    scaleFactorHeight = originalHeight * 1.0 / imgHeight;
    scaledBitmap.getPixels(intValues, 0, imgWidth, 0, 0, imgWidth, imgHeight);
    scaledBitmap.recycle();
    loadsInputToByteBuffer();
  }
}
