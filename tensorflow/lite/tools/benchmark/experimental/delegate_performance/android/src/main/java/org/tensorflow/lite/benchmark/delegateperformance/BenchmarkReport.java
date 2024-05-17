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
import java.util.List;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * Benchmark session-level report class to store the model-level {@link
 * ModelBenchmarkReportInterface} reports. It allows {@link ReportWriter} subscriptions for
 * exporting this report.
 */
final class BenchmarkReport {
  private static final String TAG = "TfLiteBenchmarkReport";
  private static final String NAME = "report";

  private final List<ReportWriter> writers = new ArrayList<>();
  private final List<ModelBenchmarkReportInterface> modelBenchmarkReports = new ArrayList<>();
  // The benchmark session-level pass status.
  // Possible values:
  // - UNKNOWN: The metric computation has not started or failed to complete.
  // - PASS: All model-level results are "PASS" or "NOT_APPLICABLE".
  // - PASS_WITH_WARNING: At least 1 "PASS_WITH_WARNING" model-level result and 0 "FAIL" model-level
  // result.
  // - FAIL: At least 1 "FAIL" model-level result.
  private BenchmarkResultType result = BenchmarkResultType.UNKNOWN;

  void addModelBenchmarkReport(ModelBenchmarkReportInterface modelBenchmarkReport) {
    modelBenchmarkReports.add(modelBenchmarkReport);
  }

  void addWriter(ReportWriter writer) {
    writers.add(writer);
  }

  void export() {
    if (result == BenchmarkResultType.UNKNOWN) {
      // The result is not computed.
      computeBenchmarkReport();
    }
    for (ReportWriter writer : writers) {
      writer.writeReport(this);
    }
  }

  // TODO(b/268338967): Use a more informative name for the report.
  String name() {
    return NAME;
  }

  List<ModelBenchmarkReportInterface> modelBenchmarkReports() {
    return Collections.unmodifiableList(modelBenchmarkReports);
  }

  BenchmarkResultType result() {
    return result;
  }

  JSONObject toJsonObject() throws JSONException {
    JSONObject jsonObject = new JSONObject();
    jsonObject.put("name", NAME);
    JSONArray jsonArray = new JSONArray();
    for (ModelBenchmarkReportInterface modelBenchmarkReport : modelBenchmarkReports) {
      jsonArray.put(modelBenchmarkReport.toJsonObject());
    }
    jsonObject.put("reports", jsonArray);
    jsonObject.put("result", result.toString());
    return jsonObject;
  }

  private void computeBenchmarkReport() {
    List<BenchmarkResultType> results = new ArrayList<>();
    for (ModelBenchmarkReportInterface modelBenchmarkReport : modelBenchmarkReports) {
      results.add(modelBenchmarkReport.result());
    }
    result = DelegatePerformanceBenchmark.aggregateResults(/* strict= */ true, results);
  }

  static BenchmarkReport create() {
    return new BenchmarkReport();
  }
}
