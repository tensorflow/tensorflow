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
import static org.tensorflow.lite.benchmark.delegateperformance.DelegatePerformanceBenchmark.checkState;

import android.util.Log;
import java.util.LinkedHashMap;
import java.util.Map;
import tflite.proto.benchmark.DelegatePerformance.BenchmarkEventType;
import tflite.proto.benchmark.DelegatePerformance.BenchmarkMetric;
import tflite.proto.benchmark.DelegatePerformance.LatencyCriteria;
import tflite.proto.benchmark.DelegatePerformance.LatencyResults;

/** Model-level latency benchmark report class. */
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
  public void addResults(LatencyResults results, TfLiteSettingsListEntry entry) {
    if (results.getEventType() != BenchmarkEventType.BENCHMARK_EVENT_TYPE_END) {
      Log.i(TAG, "The latency benchmarking is not completed successfully for " + entry.filePath());
      return;
    }
    // Use {@code LinkedHashMap} to keep the metrics insertion order.
    Map<String, Float> metrics = new LinkedHashMap<>();
    for (BenchmarkMetric metric : results.getMetricsList()) {
      if (metric != null) {
        metrics.put(metric.getName(), metric.getValue());
      }
    }
    checkState(metrics.containsKey("initialization_latency_us"));
    checkState(metrics.containsKey("warmup_latency_average_us"));
    checkState(metrics.containsKey("inference_latency_average_us"));
    metrics.put(
        "startup_overhead_latency_us",
        metrics.get("initialization_latency_us")
            + metrics.get("warmup_latency_average_us")
            - metrics.get("inference_latency_average_us"));
    rawDelegateMetrics.add(
        RawDelegateMetricsEntry.create(
            entry.tfliteSettings().delegate(), entry.filePath(), entry.isTestTarget(), metrics));
  }

  static ModelBenchmarkReport<LatencyResults> create(String modelName, LatencyCriteria criteria) {
    return new LatencyBenchmarkReport(modelName, criteria);
  }
}
