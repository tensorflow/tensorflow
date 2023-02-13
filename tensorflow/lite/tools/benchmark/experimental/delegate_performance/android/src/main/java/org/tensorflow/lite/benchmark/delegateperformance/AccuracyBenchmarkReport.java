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
import java.util.LinkedHashMap;
import java.util.Map;
import tflite.BenchmarkEvent;
import tflite.BenchmarkEventType;
import tflite.BenchmarkMetric;
import tflite.BenchmarkResult;

/** Model-level accuracy benchmark report class. */
final class AccuracyBenchmarkReport extends ModelBenchmarkReport<BenchmarkEvent> {
  public static final float PASS = 0f;
  public static final float FAIL = 1f;
  private static final String TAG = "AccuracyBenchmarkReport";

  private AccuracyBenchmarkReport(String modelName) {
    super(modelName);
    Log.i(TAG, "Creating an accuracy benchmark report for " + modelName);
    // Adds a regression threshold for "ok" here to make sure that "ok" will be consumed during
    // metric computation.
    // TODO(b/267313326): replace the mitigation with a proper accuracy benchmark criteria.
    maxRegressionPercentageAllowed.put("ok", 0f);
  }

  /**
   * Parses {@link BenchmarkEvent} into the unified {@link RawDelegateMetricsEntry} format for
   * further processing.
   *
   * <p>TODO(b/250877013): Add concrete implementation to this method.
   */
  @Override
  public void addResults(BenchmarkEvent event, TfLiteSettingsListEntry entry) {
    if (event == null || event.eventType() != BenchmarkEventType.END || event.result() == null) {
      Log.i(TAG, "The accuracy benchmarking is not completed successfully for " + entry.filePath());
      return;
    }
    Map<String, Float> metrics = new LinkedHashMap<>();
    BenchmarkResult accuracyResults = event.result();
    for (int i = 0; i < accuracyResults.metricsLength(); i++) {
      BenchmarkMetric metric = accuracyResults.metrics(i);
      checkNotNull(metric);
      if (metric.valuesLength() == 0) {
        Log.i(TAG, "The metric " + metric.name() + " is empty. Skipping to the next metric.");
        continue;
      }
      String metricName = metric.name();
      float metricValue = metric.values(0);
      if (metric.valuesLength() > 1) {
        // TODO(b/267765648): consider updating the metric aggregation logic.
        metricName += "(average)";
        float sum = 0f;
        for (int j = 0; j < metric.valuesLength(); j++) {
          sum += metric.values(j);
        }
        metricValue = sum / metric.valuesLength();
      }
      metrics.put(metricName, metricValue);
    }
    // The value of {@code ok} is set to {@code 0} when the delegate-under-test has passed the
    // accuracy checks performed by MiniBenchmark. Otherwise, the value of {@code ok} is set to
    // {@code 1}.
    // TODO(b/267313326): replace the mitigation with a proper accuracy benchmark criteria.
    metrics.put("ok", accuracyResults.ok() ? PASS : FAIL);
    metrics.put("max_memory_kb", (float) accuracyResults.maxMemoryKb());
    rawDelegateMetrics.add(
        RawDelegateMetricsEntry.create(
            entry.tfliteSettings().delegate(), entry.filePath(), entry.isTestTarget(), metrics));
  }

  static ModelBenchmarkReport<BenchmarkEvent> create(String modelName) {
    return new AccuracyBenchmarkReport(modelName);
  }
}
