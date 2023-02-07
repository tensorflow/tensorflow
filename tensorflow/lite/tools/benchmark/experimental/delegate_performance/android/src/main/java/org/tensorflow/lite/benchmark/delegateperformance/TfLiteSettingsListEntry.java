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

import android.util.Log;
import java.util.HashMap;
import tflite.BenchmarkEvent;
import tflite.BenchmarkEventType;
import tflite.BenchmarkMetric;
import tflite.BenchmarkResult;
import tflite.TFLiteSettings;
import tflite.proto.benchmark.DelegatePerformance;
import tflite.proto.benchmark.DelegatePerformance.LatencyResults;

/**
 * Helper class for store the data before and after benchmark runs. Parses both latency and accuracy
 * benchmark results.
 */
final class TfLiteSettingsListEntry {
  private static final String TAG = "TfLiteSettingsListEntry";

  private final TFLiteSettings tfliteSettings;
  private final String filePath;

  private HashMap<String, Float> metrics = new HashMap<>();

  private TfLiteSettingsListEntry(TFLiteSettings tfliteSettings, String filePath) {
    if (tfliteSettings == null) {
      throw new NullPointerException("Null tfliteSettings");
    }
    this.tfliteSettings = tfliteSettings;
    if (filePath == null) {
      throw new NullPointerException("Null filePath");
    }
    this.filePath = filePath;
  }

  TFLiteSettings tfliteSettings() {
    return tfliteSettings;
  }

  String filePath() {
    return filePath;
  }

  void setLatencyResults(LatencyResults latencyResults) {
    if (latencyResults.getEventType()
        != DelegatePerformance.BenchmarkEventType.BENCHMARK_EVENT_TYPE_END) {
      Log.i(TAG, "The latency benchmarking is not completed successfully for " + filePath);
      return;
    }
    for (DelegatePerformance.BenchmarkMetric metric : latencyResults.getMetricsList()) {
      if (metric != null) {
        metrics.put(metric.getName(), metric.getValue());
      }
    }
  }

  void setAccuracyResults(BenchmarkEvent accuracyEvent) {
    if (accuracyEvent == null
        || accuracyEvent.eventType() != BenchmarkEventType.END
        || accuracyEvent.result() == null) {
      Log.i(TAG, "The accuracy benchmarking is not completed successfully for " + filePath);
      return;
    }
    BenchmarkResult accuracyResults = accuracyEvent.result();
    for (int i = 0; i < accuracyResults.metricsLength(); i++) {
      BenchmarkMetric metric = accuracyResults.metrics(i);
      if (metric == null || metric.valuesLength() == 0) {
        continue;
      }
      String metricName = metric.name();
      float metricValue = metric.values(0);
      if (metric.valuesLength() > 1) {
        metricName += "(average)";
        float sum = 0f;
        for (int j = 0; j < metric.valuesLength(); j++) {
          sum += metric.values(j);
        }
        metricValue = sum / metric.valuesLength();
      }
      metrics.put(metricName, metricValue);
    }
    metrics.put("ok", accuracyResults.ok() ? 1.0f : 0.0f);
    metrics.put("max_memory_kb", (float) accuracyResults.maxMemoryKb());
  }

  HashMap<String, Float> metrics() {
    return metrics;
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
        + "metrics="
        + metrics
        + "}";
  }

  static TfLiteSettingsListEntry create(TFLiteSettings tfliteSettings, String filePath) {
    return new TfLiteSettingsListEntry(tfliteSettings, filePath);
  }
}
