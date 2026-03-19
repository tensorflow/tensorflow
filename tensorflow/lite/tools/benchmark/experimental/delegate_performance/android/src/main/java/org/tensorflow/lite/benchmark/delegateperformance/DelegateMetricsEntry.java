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

import java.util.Collections;
import java.util.Map;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * Helper class to store the performance results that were computed by {@link ModelBenchmarkReport}
 * from the raw performance metrics.
 *
 * <p>The computation compares the performance metrics of a reference delegate and the test target
 * delegate. Then it checks the performance regressions (if any) with the criteria to generate
 * metric-level and delegate-level results.
 *
 * <p>An instance of {@link DelegateMetricsEntry} corresponds with each tested model and each pair
 * reference delegate and the test target delegate.
 */
final class DelegateMetricsEntry {
  // Identifies the delegate involved in the computation.
  // Format: "DELEGATE_TYPE (PATH_TO_DELEGATE_SETTINGS_FILE)"
  // TODO(b/267488243): use the delegate name instead of delegate type.
  private final String delegateIdentifier;
  /** Map from performance metric names to {@link MetricsEntry}. */
  private final Map<String, MetricsEntry> metrics;
  // The delegate-level pass status. The value computation involves computing the test target
  // delegate performance regressions and checking if the regression thresholds, specified in the
  // criteria, are breached.
  // Possible values:
  // - NOT_APPLICABLE: The reference delegate is the test target delegate.
  // 1. When the test target delegate type is the same as the reference delegate.
  //    - PASS: All performance metrics of the test target delegate are better than
  //    or equal to the reference delegate.
  //    - PASS_WITH_WARNING: No regression thresholds are breached.
  //    - FAIL: At least 1 regression threshold is breached.
  // 2. When the test target delegate type is different from the reference delegate.
  //    - PASS: All performance metrics of the test target delegate are better or
  //    equal to the reference delegate.
  //    - PASS_WITH_WARNING: At least 1 regression threshold is not breached.
  //    - FAIL: All regression thresholds are breached.
  private final BenchmarkResultType result;
  private final boolean isTestTarget;
  /**
   * The value is {@code true} when the results are generated from comparing two delegates of the
   * same type. Otherwise, the value is {@code false}.
   */
  private final boolean isStrictCriteria;

  private DelegateMetricsEntry(
      String delegateIdentifier,
      Map<String, MetricsEntry> metrics,
      BenchmarkResultType result,
      boolean isTestTarget,
      boolean isStrictCriteria) {
    this.delegateIdentifier = delegateIdentifier;
    this.metrics = metrics;
    this.result = result;
    this.isTestTarget = isTestTarget;
    this.isStrictCriteria = isStrictCriteria;
  }

  /** Returns an identifer to the delegate involved in the computation. */
  String delegateIdentifier() {
    return delegateIdentifier;
  }

  /** Returns an unmodifiable map from performance metric names to {@link MetricsEntry}. */
  Map<String, MetricsEntry> metrics() {
    return Collections.unmodifiableMap(metrics);
  }

  /** Returns the delegate-level pass status. */
  BenchmarkResultType result() {
    return result;
  }

  boolean isTestTarget() {
    return isTestTarget;
  }

  /**
   * Returns {@code true} when the results are generated from comparing two delegates of the same
   * type. Otherwise, returns {@code false}.
   */
  boolean isStrictCriteria() {
    return isStrictCriteria;
  }

  JSONObject toJsonObject() throws JSONException {
    JSONObject jsonObject = new JSONObject();
    jsonObject.put("delegate_identifier", delegateIdentifier);
    jsonObject.put("result", result.toString());
    JSONObject metricsObject = new JSONObject();
    for (Map.Entry<String, MetricsEntry> entry : metrics.entrySet()) {
      metricsObject.put(entry.getKey(), entry.getValue().toJsonObject());
    }
    jsonObject.put("metrics", metricsObject);
    jsonObject.put("is_test_target", isTestTarget);
    jsonObject.put("is_strict_criteria", isStrictCriteria);
    return jsonObject;
  }

  static DelegateMetricsEntry create(
      String delegateIdentifier,
      Map<String, MetricsEntry> metrics,
      BenchmarkResultType result,
      boolean isTestTarget,
      boolean isStrictCriteria) {
    return new DelegateMetricsEntry(
        delegateIdentifier, metrics, result, isTestTarget, isStrictCriteria);
  }
}
