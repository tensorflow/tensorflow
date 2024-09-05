/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite.benchmark.delegateperformance;

import static org.tensorflow.lite.benchmark.delegateperformance.Preconditions.checkNotNull;

import tflite.TFLiteSettings;

/** Helper class for store the data before and after benchmark runs. */
final class TfLiteSettingsListEntry {
  private static final String TAG = "TfLiteSettingsListEntry";

  private final TFLiteSettings tfliteSettings;
  private final String filePath;
  private final boolean isTestTarget;

  private TfLiteSettingsListEntry(
      TFLiteSettings tfliteSettings, String filePath, boolean isTestTarget) {
    checkNotNull(tfliteSettings);
    checkNotNull(filePath);
    this.tfliteSettings = tfliteSettings;
    this.filePath = filePath;
    this.isTestTarget = isTestTarget;
  }

  TFLiteSettings tfliteSettings() {
    return tfliteSettings;
  }

  String filePath() {
    return filePath;
  }

  boolean isTestTarget() {
    return isTestTarget;
  }

  @Override
  public String toString() {
    return "TfLiteSettingsListEntry{"
        // TODO(b/265268620): Dump the entire TFLiteSettings buffer.
        + "delegate="
        + tfliteSettings.delegate()
        + ", "
        + "filePath="
        + filePath
        + ", "
        + "isTestTarget="
        + isTestTarget
        + "}";
  }

  static TfLiteSettingsListEntry create(
      TFLiteSettings tfliteSettings, String filePath, boolean isTestTarget) {
    return new TfLiteSettingsListEntry(tfliteSettings, filePath, isTestTarget);
  }
}
