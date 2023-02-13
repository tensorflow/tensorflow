/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

import android.util.Log;
import tflite.BenchmarkEvent;

/**
 * Model-level accuracy benchmark report class.
 *
 * <p>TODO(b/250877013): Add concrete implementation to this class.
 */
final class AccuracyBenchmarkReport extends ModelBenchmarkReport<BenchmarkEvent> {
  private static final String TAG = "AccuracyBenchmarkReport";

  private AccuracyBenchmarkReport(String modelName) {
    super(modelName);
    Log.i(TAG, "Creating an accuracy benchmark report for " + modelName);
  }

  /**
   * Parses {@link BenchmarkEvent} into the unified {@link RawDelegateMetricsEntry} format for
   * further processing.
   *
   * <p>TODO(b/250877013): Add concrete implementation to this method.
   */
  @Override
  public void addResults(BenchmarkEvent results, TfLiteSettingsListEntry entry) {}

  static ModelBenchmarkReport<BenchmarkEvent> create(String modelName) {
    return new AccuracyBenchmarkReport(modelName);
  }
}
