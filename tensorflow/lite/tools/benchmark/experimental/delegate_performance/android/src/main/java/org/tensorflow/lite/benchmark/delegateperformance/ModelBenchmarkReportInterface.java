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

import java.util.List;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * An interface to handle metric computations and exporting results. It is used in {@link
 * ModelBenchmarkReport}.
 */
public interface ModelBenchmarkReportInterface {
  /**
   * Converts the list of {@link RawDelegateMetricsEntry}, the raw performance results collected
   * from the native layer, into a list of {@link DelegateMetricsEntry}.
   *
   * <p>The computation compares the test target delegate with every reference delegate by computing
   * the regression of the raw performance results. The list of {@link DelegateMetricsEntry} stores
   * the calculated results for all pairs. {@link ModelBenchmarkReport} stores both the raw
   * performance results and the processed performance results.
   */
  void computeModelReport();

  /** Returns the model name. */
  String modelName();

  /** Returns the list of processed performance results for all delegates. */
  List<DelegateMetricsEntry> processedDelegateMetrics();

  /**
   * Returns the model-level pass status.
   *
   * <p>Possible return values:
   *
   * <ul>
   *   <li>1. PASS: All delegate-level results are "PASS" or "NOT_APPLICABLE".
   *   <li>2. PASS_WITH_WARNING: At least 1 "PASS_WITH_WARNING" delegate-level result and 0 "FAIL"
   *       delegate-level result.
   *   <li>3. FAIL: At least 1 "FAIL" delegate-level result.
   * </ul>
   */
  BenchmarkResultType result();

  /** Serializes the report and returns a {@code JSONObject}. */
  JSONObject toJsonObject() throws JSONException;
}
