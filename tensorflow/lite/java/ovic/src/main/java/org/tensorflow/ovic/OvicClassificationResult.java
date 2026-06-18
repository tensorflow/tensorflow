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
public class OvicClassificationResult {

  /** Top K classes and probabilities. */
  public final ArrayList<String> topKClasses;

  public final ArrayList<Float> topKProbs;
  public final ArrayList<Integer> topKIndices;

  /** Latency (ms). */
  public Long latencyMilli;

  /** Latency (ns). */
  public Long latencyNano;

  OvicClassificationResult() {
    topKClasses = new ArrayList<>();
    topKProbs = new ArrayList<>();
    topKIndices = new ArrayList<>();
    latencyMilli = -1L;
    latencyNano = -1L;
  }

  @Override
  public String toString() {
    String textToShow = latencyMilli + "ms";
    textToShow += "\n" + latencyNano + "ns";
    for (int k = 0; k < topKProbs.size(); ++k) {
      textToShow +=
          "\nPrediction ["
              + k
              + "] = Class "
              + topKIndices.get(k)
              + " ("
              + topKClasses.get(k)
              + ") : "
              + topKProbs.get(k);
    }
    return textToShow;
  }
}
