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

import java.util.ArrayList;

/** Result class for inference run on a single image. */
public class OvicSingleImageResult {

  /** Top K classes and probabilities. */
  public ArrayList<String> topKClasses;
  public ArrayList<Float> topKProbs;
  public ArrayList<Integer> topKIndices;

  /** Latency (ms). */
  public Long latency;

  OvicSingleImageResult() {
    topKClasses = new ArrayList<>();
    topKProbs = new ArrayList<>();
    topKIndices = new ArrayList<>();
    latency = -1L;
  }

  @Override
  public String toString() {
    String textToShow = latency + "ms";
    for (int k = 0; k < topKProbs.size(); ++k) {
      textToShow +=
          "\nPrediction ["
              + k
              + "] = Class "
              + Integer.toString(topKIndices.get(k))
              + " ("
              + topKClasses.get(k)
              + ") : "
              + Float.toString(topKProbs.get(k));
    }
    return textToShow;
  }

}
