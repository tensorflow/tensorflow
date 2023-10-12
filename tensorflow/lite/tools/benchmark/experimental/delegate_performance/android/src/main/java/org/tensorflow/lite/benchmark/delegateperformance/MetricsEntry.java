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

import org.json.JSONException;
import org.json.JSONObject;

/**
 * Helper class to store the metric value, the test target delegate regression compared with the
 * reference delegate and the metric-level results based on the criteria.
 *
 * <p>An instance of {@link MetricsEntry} corresponds with each tested model metric and each pair of
 * test target delegate and reference delegate.
 *
 * <p>TODO(b/267429312): use AutoValue here to simplify the source code.
 */
final class MetricsEntry {
  private final double value;
  // The test target delegate's performance regression compared against the reference delegate.
  // Example value: "5%".
  private final String regression;
  // The metric pass status after checking the regression with the criteria.
  // Possible values:
  // - NOT_APPLICABLE: This performance metric is not involved in the criteria.
  // - PASS: The performance metric of the test target delegate are better or equal to the reference
  // delegate.
  // - PASS_WITH_WARNING: The regression doesn't breach the threshold specified in the criteria.
  // - FAIL: The regression breaches the threshold specified in the criteria.
  private final BenchmarkResultType result;

  private MetricsEntry(double value, String regression, BenchmarkResultType result) {
    this.value = value;
    this.regression = regression;
    this.result = result;
  }

  double value() {
    return value;
  }

  String regression() {
    return regression;
  }

  BenchmarkResultType result() {
    return result;
  }

  JSONObject toJsonObject() throws JSONException {
    JSONObject jsonObject = new JSONObject();
    jsonObject.put("value", value);
    jsonObject.put("regression", regression);
    jsonObject.put("result", result.toString());
    return jsonObject;
  }

  static MetricsEntry create(double value, String regression, BenchmarkResultType result) {
    return new MetricsEntry(value, regression, result);
  }
}
