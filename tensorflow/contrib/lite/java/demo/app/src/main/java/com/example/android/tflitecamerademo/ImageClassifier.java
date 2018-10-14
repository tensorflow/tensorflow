/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

package com.example.android.tflitecamerademo;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.text.style.ForegroundColorSpan;
import android.text.style.RelativeSizeSpan;
import android.util.Log;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import org.tensorflow.lite.Interpreter;

/**
 * Classifies images with Tensorflow Lite.
 */
public abstract class ImageClassifier {
  // Display preferences
  private static final float GOOD_PROB_THRESHOLD = 0.3f;
  private static final int SMALL_COLOR = 0xffddaa88;

  /** Tag for the {@link Log}. */
  private static final String TAG = "TfLiteCameraDemo";

  /** Number of results to show in the UI. */
  private static final int RESULTS_TO_SHOW = 3;

  /** Dimensions of inputs. */
  private static final int DIM_BATCH_SIZE = 1;

  private static final int DIM_PIXEL_SIZE = 3;

  /** Preallocated buffers for storing image data in. */
  private int[] intValues = new int[getImageSizeX() * getImageSizeY()];

  /** Options for configuring the Interpreter. */
  private final Interpreter.Options tfliteOptions = new Interpreter.Options();

  /** The loaded TensorFlow Lite model. */
  private MappedByteBuffer tfliteModel;

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  protected Interpreter tflite;

  /** Labels corresponding to the output of the vision model. */
  private List<String> labelList;

  /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
  protected ByteBuffer imgData = null;

  /** multi-stage low pass filter * */
  private float[][] filterLabelProbArray = null;

  private static final int FILTER_STAGES = 3;
  private static final float FILTER_FACTOR = 0.4f;

  private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
      new PriorityQueue<>(
          RESULTS_TO_SHOW,
          new Comparator<Map.Entry<String, Float>>() {
            @Override
            public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
              return (o1.getValue()).compareTo(o2.getValue());
            }
          });

  /** Initializes an {@code ImageClassifier}. */
  ImageClassifier(Activity activity) throws IOException {
    tfliteModel = loadModelFile(activity);
    tflite = new Interpreter(tfliteModel, tfliteOptions);
    labelList = loadLabelList(activity);
    imgData =
        ByteBuffer.allocateDirect(
            DIM_BATCH_SIZE
                * getImageSizeX()
                * getImageSizeY()
                * DIM_PIXEL_SIZE
                * getNumBytesPerChannel());
    imgData.order(ByteOrder.nativeOrder());
    filterLabelProbArray = new float[FILTER_STAGES][getNumLabels()];
    Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");
  }

  /** Classifies a frame from the preview stream. */
  void classifyFrame(Bitmap bitmap, SpannableStringBuilder builder) {
    printTopKLabels(builder);

    if (tflite == null) {
      Log.e(TAG, "Image classifier has not been initialized; Skipped.");
      builder.append(new SpannableString("Uninitialized Classifier."));
    }
    convertBitmapToByteBuffer(bitmap);
    // Here's where the magic happens!!!
    long startTime = SystemClock.uptimeMillis();
    runInference();
    long endTime = SystemClock.uptimeMillis();
    Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));

    // Smooth the results across frames.
    applyFilter();

    // Print the results.
    long duration = endTime - startTime;
    SpannableString span = new SpannableString(duration + " ms");
    span.setSpan(new ForegroundColorSpan(android.graphics.Color.LTGRAY), 0, span.length(), 0);
    builder.append(span);
  }

  void applyFilter() {
    int numLabels = getNumLabels();

    // Low pass filter `labelProbArray` into the first stage of the filter.
    for (int j = 0; j < numLabels; ++j) {
      filterLabelProbArray[0][j] +=
          FILTER_FACTOR * (getProbability(j) - filterLabelProbArray[0][j]);
    }
    // Low pass filter each stage into the next.
    for (int i = 1; i < FILTER_STAGES; ++i) {
      for (int j = 0; j < numLabels; ++j) {
        filterLabelProbArray[i][j] +=
            FILTER_FACTOR * (filterLabelProbArray[i - 1][j] - filterLabelProbArray[i][j]);
      }
    }

    // Copy the last stage filter output back to `labelProbArray`.
    for (int j = 0; j < numLabels; ++j) {
      setProbability(j, filterLabelProbArray[FILTER_STAGES - 1][j]);
    }
  }

  private void recreateInterpreter() {
    if (tflite != null) {
      tflite.close();
      tflite = new Interpreter(tfliteModel, tfliteOptions);
    }
  }

  public void setUseNNAPI(Boolean nnapi) {
    tfliteOptions.setUseNNAPI(nnapi);
    recreateInterpreter();
  }

  public void setNumThreads(int numThreads) {
    tfliteOptions.setNumThreads(numThreads);
    recreateInterpreter();
  }

  /** Closes tflite to release resources. */
  public void close() {
    tflite.close();
    tflite = null;
    tfliteModel = null;
  }

  /** Reads label list from Assets. */
  private List<String> loadLabelList(Activity activity) throws IOException {
    List<String> labelList = new ArrayList<String>();
    BufferedReader reader =
        new BufferedReader(new InputStreamReader(activity.getAssets().open(getLabelPath())));
    String line;
    while ((line = reader.readLine()) != null) {
      labelList.add(line);
    }
    reader.close();
    return labelList;
  }

  /** Memory-map the model file in Assets. */
  private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(getModelPath());
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /** Writes Image data into a {@code ByteBuffer}. */
  private void convertBitmapToByteBuffer(Bitmap bitmap) {
    if (imgData == null) {
      return;
    }
    imgData.rewind();
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    // Convert the image to floating point.
    int pixel = 0;
    long startTime = SystemClock.uptimeMillis();
    for (int i = 0; i < getImageSizeX(); ++i) {
      for (int j = 0; j < getImageSizeY(); ++j) {
        final int val = intValues[pixel++];
        addPixelValue(val);
      }
    }
    long endTime = SystemClock.uptimeMillis();
    Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
  }

  /** Prints top-K labels, to be shown in UI as the results. */
  private void printTopKLabels(SpannableStringBuilder builder) {
    for (int i = 0; i < getNumLabels(); ++i) {
      sortedLabels.add(
          new AbstractMap.SimpleEntry<>(labelList.get(i), getNormalizedProbability(i)));
      if (sortedLabels.size() > RESULTS_TO_SHOW) {
        sortedLabels.poll();
      }
    }

    final int size = sortedLabels.size();
    for (int i = 0; i < size; i++) {
      Map.Entry<String, Float> label = sortedLabels.poll();
      SpannableString span =
          new SpannableString(String.format("%s: %4.2f\n", label.getKey(), label.getValue()));
      int color;
      // Make it white when probability larger than threshold.
      if (label.getValue() > GOOD_PROB_THRESHOLD) {
        color = android.graphics.Color.WHITE;
      } else {
        color = SMALL_COLOR;
      }
      // Make first item bigger.
      if (i == size - 1) {
        float sizeScale = (i == size - 1) ? 1.25f : 0.8f;
        span.setSpan(new RelativeSizeSpan(sizeScale), 0, span.length(), 0);
      }
      span.setSpan(new ForegroundColorSpan(color), 0, span.length(), 0);
      builder.insert(0, span);
    }
  }

  /**
   * Get the name of the model file stored in Assets.
   *
   * @return
   */
  protected abstract String getModelPath();

  /**
   * Get the name of the label file stored in Assets.
   *
   * @return
   */
  protected abstract String getLabelPath();

  /**
   * Get the image size along the x axis.
   *
   * @return
   */
  protected abstract int getImageSizeX();

  /**
   * Get the image size along the y axis.
   *
   * @return
   */
  protected abstract int getImageSizeY();

  /**
   * Get the number of bytes that is used to store a single color channel value.
   *
   * @return
   */
  protected abstract int getNumBytesPerChannel();

  /**
   * Add pixelValue to byteBuffer.
   *
   * @param pixelValue
   */
  protected abstract void addPixelValue(int pixelValue);

  /**
   * Read the probability value for the specified label This is either the original value as it was
   * read from the net's output or the updated value after the filter was applied.
   *
   * @param labelIndex
   * @return
   */
  protected abstract float getProbability(int labelIndex);

  /**
   * Set the probability value for the specified label.
   *
   * @param labelIndex
   * @param value
   */
  protected abstract void setProbability(int labelIndex, Number value);

  /**
   * Get the normalized probability value for the specified label. This is the final value as it
   * will be shown to the user.
   *
   * @return
   */
  protected abstract float getNormalizedProbability(int labelIndex);

  /**
   * Run inference using the prepared input in {@link #imgData}. Afterwards, the result will be
   * provided by getProbability().
   *
   * <p>This additional method is necessary, because we don't have a common base for different
   * primitive data types.
   */
  protected abstract void runInference();

  /**
   * Get the total number of labels.
   *
   * @return
   */
  protected int getNumLabels() {
    return labelList.size();
  }
}
