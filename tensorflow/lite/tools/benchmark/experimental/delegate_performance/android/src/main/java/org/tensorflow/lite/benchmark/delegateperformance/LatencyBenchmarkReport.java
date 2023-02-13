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

import static org.tensorflow.lite.benchmark.delegateperformance.DelegatePerformanceBenchmark.checkNotNull;

import android.util.Log;
import tflite.proto.benchmark.DelegatePerformance.LatencyCriteria;
import tflite.proto.benchmark.DelegatePerformance.LatencyResults;

/**
 * Model-level latency benchmark report class.
 *
 * <p>TODO(b/250877013): Add concrete implementation.
 */
final class LatencyBenchmarkReport extends ModelBenchmarkReport<LatencyResults> {
  private static final String TAG = "LatencyBenchmarkReport";

  private LatencyBenchmarkReport(String modelName, LatencyCriteria criteria) {
    super(modelName);
    checkNotNull(criteria);
    this.maxRegressionPercentageAllowed.put(
        "startup_overhead_latency_us", criteria.getStartupOverheadMaxRegressionPercentageAllowed());
    this.maxRegressionPercentageAllowed.put(
        "inference_latency_average_us",
        criteria.getAverageInferenceMaxRegressionPercentageAllowed());
    Log.i(TAG, "Creating a latency benchmark report for " + modelName);
  }

  /**
   * Parses {@link LatencyResults} into the unified {@link RawDelegateMetricsEntry} format for
   * further processing.
   */
  @Override
  public void addResults(LatencyResults results, TfLiteSettingsListEntry entry) {}

  static ModelBenchmarkReport<LatencyResults> create(String modelName, LatencyCriteria criteria) {
    return new LatencyBenchmarkReport(modelName, criteria);
  }
}
