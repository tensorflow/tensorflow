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

import static org.tensorflow.lite.benchmark.delegateperformance.DelegatePerformanceBenchmark.checkState;

import android.util.Log;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

/** Helper class for writing the final report in HTML format. */
final class HtmlWriter implements ReportWriter {
  private static final String TAG = "TfLiteHtmlWriter";

  private final String destinationFolderPath;

  private HtmlWriter(String destinationFolderPath) {
    this.destinationFolderPath = destinationFolderPath;
  }

  /** Writes the benchmark results into an HTML file. */
  @Override
  public void writeReport(BenchmarkReport report) {
    // Example output file contains an overall summary table like:
    //
    // Summary PASS
    //
    // And a detailed methods table like:
    //
    // Model   Metric           DELEGATE_TYPE (PATH) DELEGATE_TYPE (PATH) Change Status ...
    // model_1 metric_1         900                  1000                 -10%   PASS   ...
    // model_1 metric_2         ...
    // model_1 delegate_summary                                                  PASS   ...
    // model_1 model_summary    PASS
    // model_2 ...
    // ...
    StringBuilder sb = new StringBuilder();
    sb.append(destinationFolderPath).append("/").append(report.name()).append(".html");
    String filePath = sb.toString();
    Log.i(TAG, "Generating an HTML report to " + filePath);

    List<ModelBenchmarkReportInterface> modelReports = report.modelBenchmarkReports();
    checkState(!modelReports.isEmpty());

    // File header
    sb =
        new StringBuilder()
            .append("<html>\n")
            .append("<head>\n")
            .append("<title>Delegate Performance Benchmark Report</title>\n")
            .append("<style>\n")
            .append("body {\n")
            .append("  font-family: -apple-system, system-ui, BlinkMacSystemFont, \"Segoe UI\",\n")
            .append("               \"Roboto\", \"Helvetica Neue\", Arial, sans-serif;\n")
            .append("}\n")
            .append("table, th, td {\n")
            .append("  border: 1px solid black;\n")
            .append("  border-collapse: collapse;\n")
            .append("  padding: 15px;\n")
            .append("  text-align: center;\n")
            .append("}\n")
            .append("th {\n")
            .append("  background-color:LightSkyBlue\n")
            .append("}\n")
            .append("tr:nth-child(even) {\n")
            .append("  background-color: whitesmoke;\n")
            .append("}\n")
            .append(".status-PASS {\n")
            .append("  background-color: chartreuse;\n")
            .append("}\n")
            .append(".status-FAIL {\n")
            .append("  background-color: red;\n")
            .append("}\n")
            .append(".status-PASS_WITH_WARNING {\n")
            .append("  background-color: yellow;\n")
            .append("}\n")
            .append(".status-NOT_APPLICABLE {\n")
            .append("  background-color: grey;\n")
            .append("}\n")
            .append(".ls-info-box {\n")
            .append("  padding: 5px 10px;\n")
            .append("  margin: 10px;\n")
            .append("  border-style: solid;\n")
            .append("  border-left-width: 5px;\n")
            .append("  border-radius: 4px;\n")
            .append("  color: #666;\n")
            .append("  font-size: 13px;\n")
            .append("  line-height: 1.3\n")
            .append("}\n")
            .append(".ls-info-box {\n")
            .append("  background-color: #f0f0f0;\n")
            .append("  border-left-color: silver\n")
            .append("}\n")
            .append("</style>\n")
            .append("</head>\n")
            .append("<body>\n")
            // TODO(b/268338967): Use a more informative name for the report.
            .append("<h1>Delegate Performance Benchmark Report</h1>\n");
    // Summary table
    sb.append("<table>\n").append("<tr>\n").append("<td>Summary</td>\n");
    addResultCell(report.result(), /* isStrictCriteria= */ null, sb);
    sb.append("</tr>\n").append("</table>\n");

    // Heading row for the detailed metric table. It is structured as below:
    // Model, Metric, DELEGATE_TYPE (PATH), DELEGATE_TYPE (PATH), Change, Status, ...
    sb.append("<table>\n")
        .append("<thead>\n")
        .append("<tr>\n")
        .append("<th>Model</th>\n")
        .append("<th>Metric</th>\n");
    for (DelegateMetricsEntry entry : modelReports.get(0).processedDelegateMetrics()) {
      sb.append("<th>Delegate: ").append(entry.delegateIdentifier()).append("</th>\n");
      if (!entry.isTestTarget()) {
        sb.append("<th>Change (%)</th>\n").append("<th>Status</th>\n");
      }
    }
    sb.append("</thead>\n").append("<tbody>\n");
    for (ModelBenchmarkReportInterface modelReport : modelReports) {
      writerModelReport(modelReport, sb);
    }
    sb.append("</tbody>\n")
        .append("</table>\n")
        .append("<div class=\"ls-info-box\">\n")
        .append("<p>\n")
        .append(
            "When the test target delegate type is the same as the reference delegate, the checks"
                + " are more strict. Otherwise, the checks are relaxed. Please see \n")
        .append(
            "<a href=\"https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/src/main/java/org/tensorflow/lite/benchmark/delegateperformance/BenchmarkResultType.java\">BenchmarkResultType.java</a>\n")
        .append(" for the meanings of PASS, PASS_WITH_WARNING and FAIL.\n")
        .append("</p>\n")
        .append("</div>\n")
        .append("</body>\n")
        .append("</html>\n");
    try (PrintWriter writer = new PrintWriter(filePath)) {
      writer.write(sb.toString());
    } catch (IOException e) {
      Log.e(TAG, "Failed to open report file " + filePath);
    }
  }

  private void writerModelReport(ModelBenchmarkReportInterface modelReport, StringBuilder sb) {
    String modelName = modelReport.modelName();
    if (modelReport.processedDelegateMetrics().isEmpty()) {
      Log.w(TAG, "The computed metric is empty.");
    } else {
      for (String metricName : modelReport.processedDelegateMetrics().get(0).metrics().keySet()) {
        sb.append("<tr>\n<td>")
            .append(modelName)
            .append("</td>\n<td>")
            .append(metricName)
            .append("</td>\n");
        for (DelegateMetricsEntry delegateMetricsEntry : modelReport.processedDelegateMetrics()) {
          MetricsEntry metricEntry = delegateMetricsEntry.metrics().get(metricName);
          sb.append("<td>").append(metricEntry.value()).append("</td>\n");
          if (!delegateMetricsEntry.isTestTarget()) {
            sb.append("<td>").append(metricEntry.regression()).append("</td>\n");
            addResultCell(metricEntry.result(), /* isStrictCriteria= */ null, sb);
          }
        }
        sb.append("</tr>\n");
      }
      sb.append("<tr>\n<td>").append(modelName).append("</td>\n<td>delegate_summary</td>\n");
      for (DelegateMetricsEntry delegateMetricsEntry : modelReport.processedDelegateMetrics()) {
        // Position the delegate-level result correctly.
        sb.append("<td/>\n");
        if (!delegateMetricsEntry.isTestTarget()) {
          sb.append("<td/>\n");
          addResultCell(delegateMetricsEntry.result(), delegateMetricsEntry.isStrictCriteria(), sb);
        }
      }
      sb.append("</tr>\n<tr>\n<td>").append(modelName).append("</td>\n<td>model_summary</td>\n");
      addResultCell(modelReport.result(), /* isStrictCriteria= */ null, sb);
      sb.append("</tr>\n");
    }
  }

  /** Adds a colored result cell to the table. */
  private void addResultCell(
      BenchmarkResultType result, Boolean isStrictCriteria, StringBuilder sb) {
    sb.append("<td class=\"status-").append(result.name()).append("\">").append(result);
    if (isStrictCriteria != null && isStrictCriteria) {
      sb.append(" (strict)");
    }
    sb.append("</td>\n");
  }

  static ReportWriter create(String destinationFolderPath) {
    return new HtmlWriter(destinationFolderPath);
  }
}
