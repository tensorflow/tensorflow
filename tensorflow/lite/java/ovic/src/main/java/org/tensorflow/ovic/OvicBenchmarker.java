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
import android.os.SystemClock;
import android.util.Log;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;

/**
 * Base class that benchmarks image models.
 *
 * <p>===================== General workflow =======================
 *
 * <pre>{@code
 * benchmarker = new OvicBenchmarker();
 * benchmarker.getReadyToTest(labelInputStream, model);
 * while (!benchmarker.shouldStop()) {
 *   Bitmap bitmap = ...
 *   imgId = ...
 *   benchmarker.processBitmap(bitmap, imgId);
 * }
 * }</pre>
 */
public abstract class OvicBenchmarker {
  /** Tag for the {@link Log}. */
  private static final String TAG = "OvicBenchmarker";

  /** Dimensions of inputs. */
  protected static final int DIM_BATCH_SIZE = 1;

  protected static final int DIM_PIXEL_SIZE = 3;
  protected int imgHeight = 224;
  protected int imgWidth = 224;

  /* Preallocated buffers for storing image data in. */
  protected int[] intValues = null;

  /** A ByteBuffer to hold image data, to be feed into classifier as inputs. */
  protected ByteBuffer imgData = null;

  /** Total runtime in ns. */
  protected double totalRuntimeNano = 0.0;
  /** Total allowed runtime in ms. */
  protected double wallTimeNano = 20000 * 30 * 1.0e6;
  /** Record whether benchmark has started (used to skip the first image). */
  protected boolean benchmarkStarted = false;

  /**
   * Initializes an {@link OvicBenchmarker}
   *
   * @param wallTimeNano: a double number specifying the total amount of time to benchmark.
   */
  public OvicBenchmarker(double wallTimeNano) {
    benchmarkStarted = false;
    totalRuntimeNano = 0.0;
    this.wallTimeNano = wallTimeNano;
  }

  /** Return the cumulative latency of all runs so far. */
  public double getTotalRuntimeNano() {
    return totalRuntimeNano;
  }

  /** Check whether the benchmarker should stop. */
  public Boolean shouldStop() {
    if (totalRuntimeNano >= wallTimeNano) {
      Log.e(
          TAG,
          "Total runtime (ms) "
              + (totalRuntimeNano * 1.0e-6)
              + " exceeded wall-time "
              + (wallTimeNano * 1.0e-6));
      return true;
    }
    return false;
  }

  /** Abstract class for checking whether the benchmarker is ready to start processing images */
  public abstract boolean readyToTest();

  /**
   * Abstract class for getting the benchmarker ready.
   *
   * @param labelInputStream: an {@link InputStream} specifying where the list of labels should be
   *     read from.
   * @param model: a {@link MappedByteBuffer} model to benchmark.
   */
  public abstract void getReadyToTest(InputStream labelInputStream, MappedByteBuffer model);

  /**
   * Perform test on a single bitmap image.
   *
   * @param bitmap: a {@link Bitmap} image to process.
   * @param imageId: an ID uniquely representing the image.
   */
  public abstract boolean processBitmap(Bitmap bitmap, int imageId)
      throws IOException, InterruptedException;

  /** Perform test on a single bitmap image without an image ID. */
  public boolean processBitmap(Bitmap bitmap) throws IOException, InterruptedException {
    return processBitmap(bitmap, /* imageId = */ 0);
  }

  /** Returns the last inference results as string. */
  public abstract String getLastResultString();

  /**
   * Loads input buffer from intValues into ByteBuffer for the interpreter. Input buffer must be
   * loaded in intValues and output will be placed in imgData.
   */
  protected void loadsInputToByteBuffer() {
    if (imgData == null || intValues == null) {
      throw new RuntimeException("Benchmarker is not yet ready to test.");
    }
    // Convert the image to ByteBuffer.
    imgData.rewind();
    int pixel = 0;
    long startTime = SystemClock.uptimeMillis();

    for (int i = 0; i < imgHeight; ++i) {
      for (int j = 0; j < imgWidth; ++j) {
        final int pixelValue = intValues[pixel++];
        imgData.put((byte) ((pixelValue >> 16) & 0xFF));
        imgData.put((byte) ((pixelValue >> 8) & 0xFF));
        imgData.put((byte) (pixelValue & 0xFF));
      }
    }
    long endTime = SystemClock.uptimeMillis();
    Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
  }
}
