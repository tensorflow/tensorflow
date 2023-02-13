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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * Model-level benchmark report class to be extended by {@link AccuracyBenchmarkReport} and {@link
 * LatencyBenchmarkReport}.
 *
 * <p>This class contains helper functions to convert the raw performance results from the native
 * layer into metric-level, delegate-level and model-level pass status values.
 *
 * <p>TODO(b/250877013): Add concrete implementation.
 */
public abstract class ModelBenchmarkReport<ResultsT> implements ModelBenchmarkReportInterface {
  private static final String TAG = "ModelBenchmarkReport";

  protected final String modelName;
  /* Map from performance metric names to the maximum regression percentage thresholds allowed. */
  protected final Map<String, Float> maxRegressionPercentageAllowed = new HashMap<>();
  /**
   * List of {@link RawDelegateMetricsEntry}, which stores delegate-level performance results
   * collected from the native layer.
   */
  protected final List<RawDelegateMetricsEntry> rawDelegateMetrics = new ArrayList<>();
  /**
   * List of {@link DelegateMetricsEntry}, which stores delegate-level performance results computed
   * by {@link #computeModelReport()}.
   */
  protected final List<DelegateMetricsEntry> processedDelegateMetrics = new ArrayList<>();
  /** Model-level pass status. The field will be updated by {@link #computeModelReport()}. */
  protected final BenchmarkResultType result = BenchmarkResultType.UNKNOWN;

  protected ModelBenchmarkReport(String modelName) {
    this.modelName = modelName;
  }

  /**
   * Parses accuracy or latency results into the unified {@link RawDelegateMetricsEntry} format for
   * further processing.
   */
  public abstract void addResults(ResultsT results, TfLiteSettingsListEntry entry);

  @Override
  public String modelName() {
    return modelName;
  }

  @Override
  public List<DelegateMetricsEntry> processedDelegateMetrics() {
    return Collections.unmodifiableList(processedDelegateMetrics);
  }

  @Override
  public BenchmarkResultType result() {
    return result;
  }

  /**
   * Converts the prepopulated list of {@link RawDelegateMetricsEntry}, the raw performance results
   * collected from the native layer, into a list of {@link DelegateMetricsEntry}.
   *
   * <p>Note: {@link #addResults(ResultsT, TfLiteSettingsListEntry)} should be called to populate
   * the list of {@link RawDelegateMetricsEntry} before calling this method.
   *
   * <p>TODO(b/268595172): Remove the above precondition.
   *
   * <p>TODO(b/250877013): Add concrete implementation to this method.
   */
  @Override
  public void computeModelReport() {}

  @Override
  public JSONObject toJsonObject() throws JSONException {
    JSONObject jsonObject = new JSONObject();
    jsonObject.put("model", modelName);
    jsonObject.put("result", result.toString());
    JSONArray processedDelegateMetricsArray = new JSONArray();
    for (DelegateMetricsEntry entry : processedDelegateMetrics) {
      processedDelegateMetricsArray.put(entry.toJsonObject());
    }
    jsonObject.put("metrics", processedDelegateMetricsArray);
    JSONArray rawDelegateMetricsArray = new JSONArray();
    for (RawDelegateMetricsEntry entry : rawDelegateMetrics) {
      rawDelegateMetricsArray.put(entry.toJsonObject());
    }
    jsonObject.put("raw_metrics", rawDelegateMetricsArray);
    JSONObject maxRegressionPercentageAllowedObject = new JSONObject();
    for (Map.Entry<String, Float> entry : maxRegressionPercentageAllowed.entrySet()) {
      maxRegressionPercentageAllowedObject.put(entry.getKey(), entry.getValue());
    }
    jsonObject.put("max_regression_percentage_allowed", maxRegressionPercentageAllowedObject);
    return jsonObject;
  }
}
