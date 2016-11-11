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
import android.os.Trace;
import android.util.Log;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/** A classifier specialized to label images using TensorFlow. */
public class TensorFlowImageClassifier implements Classifier {
  static {
    System.loadLibrary("tensorflow_demo");
  }

  private static final String TAG = "TensorFlowImageClassifier";

  // Only return this many results with at least this confidence.
  private static final int MAX_RESULTS = 3;
  private static final float THRESHOLD = 0.1f;

  // Config values.
  private String inputName;
  private String outputName;
  private int inputSize;
  private int imageMean;
  private float imageStd;

  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();
  private int[] intValues;
  private float[] floatValues;
  private float[] outputs;
  private String[] outputNames;

  private TensorFlowInferenceInterface inferenceInterface;

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   * @param numClasses The number of classes output by the model.
   * @param inputSize The input size. A square image of inputSize x inputSize is assumed.
   * @param imageMean The assumed mean of the image values.
   * @param imageStd The assumed std of the image values.
   * @param inputName The label of the image input node.
   * @param outputName The label of the output node.
   * @return The native return value, 0 indicating success.
   * @throws IOException
   */
  public int initializeTensorFlow(
      AssetManager assetManager,
      String modelFilename,
      String labelFilename,
      int numClasses,
      int inputSize,
      int imageMean,
      float imageStd,
      String inputName,
      String outputName) throws IOException {
    this.inputName = inputName;
    this.outputName = outputName;

    // Read the label names into memory.
    // TODO(andrewharp): make this handle non-assets.
    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    Log.i(TAG, "Reading labels from: " + actualFilename);
    BufferedReader br = null;
    br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
    String line;
    while ((line = br.readLine()) != null) {
      labels.add(line);
    }
    br.close();
    Log.i(TAG, "Read " + labels.size() + ", " + numClasses + " specified");

    this.inputSize = inputSize;
    this.imageMean = imageMean;
    this.imageStd = imageStd;

    // Pre-allocate buffers.
    outputNames = new String[] {outputName};
    intValues = new int[inputSize * inputSize];
    floatValues = new float[inputSize * inputSize * 3];
    outputs = new float[numClasses];

    inferenceInterface = new TensorFlowInferenceInterface();

    return inferenceInterface.initializeTensorFlow(assetManager, modelFilename);
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
      final int val = intValues[i];
      floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
      floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
      floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
    }
    Trace.endSection();

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
    inferenceInterface.readNodeFloat(outputName, outputs);
    Trace.endSection();

    // Find the best classifications.
    PriorityQueue<Recognition> pq = new PriorityQueue<Recognition>(3,
        new Comparator<Recognition>() {
          @Override
          public int compare(Recognition lhs, Recognition rhs) {
            // Intentionally reversed to put high confidence at the head of the queue.
            return Float.compare(rhs.getConfidence(), lhs.getConfidence());
          }
        });
    for (int i = 0; i < outputs.length; ++i) {
      if (outputs[i] > THRESHOLD) {
        pq.add(new Recognition(
            "" + i, labels.get(i), outputs[i], null));
      }
    }
    final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
    int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
    for (int i = 0; i < recognitionsSize; ++i) {
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
