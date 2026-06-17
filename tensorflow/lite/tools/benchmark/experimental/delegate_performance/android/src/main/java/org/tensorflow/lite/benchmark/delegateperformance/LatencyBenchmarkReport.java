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

import static java.lang.Math.max;
import static org.tensorflow.lite.benchmark.delegateperformance.Preconditions.checkNotNull;
import static org.tensorflow.lite.benchmark.delegateperformance.Preconditions.checkState;

import android.util.Log;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import tflite.proto.benchmark.DelegatePerformance.BenchmarkEventType;
import tflite.proto.benchmark.DelegatePerformance.BenchmarkMetric;
import tflite.proto.benchmark.DelegatePerformance.LatencyCriteria;
import tflite.proto.benchmark.DelegatePerformance.LatencyResults;

/** Model-level latency benchmark report class. */
final class LatencyBenchmarkReport extends ModelBenchmarkReport {
  private static final String TAG = "LatencyBenchmarkReport";

  private LatencyBenchmarkReport(
      String modelName,
      List<RawDelegateMetricsEntry> rawDelegateMetricsEntries,
      LatencyCriteria criteria) {
    super(modelName);
    checkNotNull(criteria);
    this.maxRegressionPercentageAllowed.put(
        "startup_overhead_latency_us",
        (double) criteria.getStartupOverheadMaxRegressionPercentageAllowed());
    this.maxRegressionPercentageAllowed.put(
        "inference_latency_average_us",
        (double) criteria.getAverageInferenceMaxRegressionPercentageAllowed());
    Log.i(TAG, "Creating a latency benchmark report for " + modelName);
    computeModelReport(rawDelegateMetricsEntries);
  }

  /**
   * Parses {@link LatencyResults} into the unified {@link RawDelegateMetricsEntry} format for
   * further processing.
   */
  public static RawDelegateMetricsEntry parseResults(
      LatencyResults results, TfLiteSettingsListEntry entry) {
    // Use {@code LinkedHashMap} to keep the metrics insertion order.
    Map<String, Double> metrics = new LinkedHashMap<>();
    if (results.getEventType() != BenchmarkEventType.BENCHMARK_EVENT_TYPE_END) {
      Log.w(TAG, "The latency benchmarking is not completed successfully for " + entry.filePath());
      return RawDelegateMetricsEntry.create(
          entry.tfliteSettings().delegate(), entry.filePath(), entry.isTestTarget(), metrics);
    }
    for (BenchmarkMetric metric : results.getMetricsList()) {
      if (metric != null) {
        metrics.put(metric.getName(), (double) metric.getValue());
      }
    }
    checkState(metrics.containsKey("initialization_latency_us"));
    checkState(metrics.containsKey("warmup_latency_average_us"));
    checkState(metrics.containsKey("inference_latency_average_us"));
    metrics.put(
        "startup_overhead_latency_us",
        max(
            metrics.get("initialization_latency_us")
                + metrics.get("warmup_latency_average_us")
                - metrics.get("inference_latency_average_us"),
            // The average warmup latency is generally assumed to be greater than the average
            // inference latency. Therefore, it is highly unusual for the sum of the initialization
            // latency and the average warmup latency to be less than the average inference latency.
            // In order to ensure that the regression is computed successfully, we keep the startup
            // overhead latency at minimum 0.
            0d));
    return RawDelegateMetricsEntry.create(
        entry.tfliteSettings().delegate(), entry.filePath(), entry.isTestTarget(), metrics);
  }

  static LatencyBenchmarkReport create(
      String modelName,
      List<RawDelegateMetricsEntry> rawDelegateMetricsEntries,
      LatencyCriteria criteria) {
    return new LatencyBenchmarkReport(modelName, rawDelegateMetricsEntries, criteria);
  }
}
