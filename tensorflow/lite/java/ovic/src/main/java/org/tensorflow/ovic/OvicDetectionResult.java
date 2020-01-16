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

import java.util.ArrayList;

/** Result class for inference run on a single image. */
public class OvicDetectionResult {

  // Top K classes and probabilities.
  public final ArrayList<BoundingBox> detections;
  // Latency (ms).
  public Long latency = -1L;
  // id of the image.
  public int id = -1;
  // Number of valid detections (separately maintained, maybe different from detections.size()).
  public int count = 0;

  // Create OvicDetectionResult object with pre-filled capacity. Note that detections.size() will
  // be equal to capacity after this call.
  OvicDetectionResult(int capacity) {
    detections = new ArrayList<BoundingBox>(capacity);
    for (int i = 0; i < capacity; i++) {
      detections.add(new BoundingBox(-1.0f, -1.0f, -1.0f, -1.0f, -1, -1.0f));
    }
  }

  public void resetTo(Long latency, int id) {
    count = 0;
    this.latency = latency;
    this.id = id;
  }

  public void addBox(float x1, float y1, float x2, float y2, int category, float score) {
    detections.get(count).x1 = x1;
    detections.get(count).y1 = y1;
    detections.get(count).x2 = x2;
    detections.get(count).y2 = y2;
    detections.get(count).category = category;
    detections.get(count).score = score;
    count += 1;
  }

  public void scaleUp(double scaleFactorWidth, double scaleFactorHeight) {
    for (BoundingBox box : detections) {
      box.x1 = (float) (box.x1 * scaleFactorWidth);
      box.y1 = (float) (box.y1 * scaleFactorHeight);
      box.x2 = (float) (box.x2 * scaleFactorWidth);
      box.y2 = (float) (box.y2 * scaleFactorHeight);
    }
  }

  @Override
  public String toString() {
    String textToShow = latency + "ms";
    int k = 0;
    for (BoundingBox box : detections) {
      textToShow +=
          "\nPrediction ["
              + k
              + "] = Class "
              + box.category
              + " ("
              + box.x1
              + ", "
              + box.y1
              + ", "
              + box.x2
              + ", "
              + box.y2
              + ") : "
              + box.score;
      k++;
    }


    return textToShow;
  }
}
