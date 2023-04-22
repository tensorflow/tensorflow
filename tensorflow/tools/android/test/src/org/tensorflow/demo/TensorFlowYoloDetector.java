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

package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.env.SplitTimer;

/** An object detector that uses TF and a YOLO model to detect objects. */
public class TensorFlowYoloDetector implements Classifier {
  private static final Logger LOGGER = new Logger();

  // Only return this many results with at least this confidence.
  private static final int MAX_RESULTS = 5;

  private static final int NUM_CLASSES = 20;

  private static final int NUM_BOXES_PER_BLOCK = 5;

  // TODO(andrewharp): allow loading anchors and classes
  // from files.
  private static final double[] ANCHORS = {
    1.08, 1.19,
    3.42, 4.41,
    6.63, 11.38,
    9.42, 5.11,
    16.62, 10.52
  };

  private static final String[] LABELS = {
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
  };

  // Config values.
  private String inputName;
  private int inputSize;

  // Pre-allocated buffers.
  private int[] intValues;
  private float[] floatValues;
  private String[] outputNames;

  private int blockSize;

  private boolean logStats = false;

  private TensorFlowInferenceInterface inferenceInterface;

  /** Initializes a native TensorFlow session for classifying images. */
  public static Classifier create(
      final AssetManager assetManager,
      final String modelFilename,
      final int inputSize,
      final String inputName,
      final String outputName,
      final int blockSize) {
    TensorFlowYoloDetector d = new TensorFlowYoloDetector();
    d.inputName = inputName;
    d.inputSize = inputSize;

    // Pre-allocate buffers.
    d.outputNames = outputName.split(",");
    d.intValues = new int[inputSize * inputSize];
    d.floatValues = new float[inputSize * inputSize * 3];
    d.blockSize = blockSize;

    d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

    return d;
  }

  private TensorFlowYoloDetector() {}

  private float expit(final float x) {
    return (float) (1. / (1. + Math.exp(-x)));
  }

  private void softmax(final float[] vals) {
    float max = Float.NEGATIVE_INFINITY;
    for (final float val : vals) {
      max = Math.max(max, val);
    }
    float sum = 0.0f;
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = (float) Math.exp(vals[i] - max);
      sum += vals[i];
    }
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = vals[i] / sum;
    }
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    final SplitTimer timer = new SplitTimer("recognizeImage");

    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    for (int i = 0; i < intValues.length; ++i) {
      floatValues[i * 3 + 0] = ((intValues[i] >> 16) & 0xFF) / 255.0f;
      floatValues[i * 3 + 1] = ((intValues[i] >> 8) & 0xFF) / 255.0f;
      floatValues[i * 3 + 2] = (intValues[i] & 0xFF) / 255.0f;
    }
    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");
    inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);
    Trace.endSection();

    timer.endSplit("ready for inference");

    // Run the inference call.
    Trace.beginSection("run");
    inferenceInterface.run(outputNames, logStats);
    Trace.endSection();

    timer.endSplit("ran inference");

    // Copy the output Tensor back into the output array.
    Trace.beginSection("fetch");
    final int gridWidth = bitmap.getWidth() / blockSize;
    final int gridHeight = bitmap.getHeight() / blockSize;
    final float[] output =
        new float[gridWidth * gridHeight * (NUM_CLASSES + 5) * NUM_BOXES_PER_BLOCK];
    inferenceInterface.fetch(outputNames[0], output);
    Trace.endSection();

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

    for (int y = 0; y < gridHeight; ++y) {
      for (int x = 0; x < gridWidth; ++x) {
        for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
          final int offset =
              (gridWidth * (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5))) * y
                  + (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5)) * x
                  + (NUM_CLASSES + 5) * b;

          final float xPos = (x + expit(output[offset + 0])) * blockSize;
          final float yPos = (y + expit(output[offset + 1])) * blockSize;

          final float w = (float) (Math.exp(output[offset + 2]) * ANCHORS[2 * b + 0]) * blockSize;
          final float h = (float) (Math.exp(output[offset + 3]) * ANCHORS[2 * b + 1]) * blockSize;

          final RectF rect =
              new RectF(
                  Math.max(0, xPos - w / 2),
                  Math.max(0, yPos - h / 2),
                  Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                  Math.min(bitmap.getHeight() - 1, yPos + h / 2));
          final float confidence = expit(output[offset + 4]);

          int detectedClass = -1;
          float maxClass = 0;

          final float[] classes = new float[NUM_CLASSES];
          for (int c = 0; c < NUM_CLASSES; ++c) {
            classes[c] = output[offset + 5 + c];
          }
          softmax(classes);

          for (int c = 0; c < NUM_CLASSES; ++c) {
            if (classes[c] > maxClass) {
              detectedClass = c;
              maxClass = classes[c];
            }
          }

          final float confidenceInClass = maxClass * confidence;
          if (confidenceInClass > 0.01) {
            LOGGER.i(
                "%s (%d) %f %s", LABELS[detectedClass], detectedClass, confidenceInClass, rect);
            pq.add(new Recognition("" + offset, LABELS[detectedClass], confidenceInClass, rect));
          }
        }
      }
    }
    timer.endSplit("decoded results");

    final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
    for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
      recognitions.add(pq.poll());
    }
    Trace.endSection(); // "recognizeImage"

    timer.endSplit("processed results");

    return recognitions;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {
    this.logStats = logStats;
  }

  @Override
  public String getStatString() {
    return inferenceInterface.getStatString();
  }

  @Override
  public void close() {
    inferenceInterface.close();
  }
}
