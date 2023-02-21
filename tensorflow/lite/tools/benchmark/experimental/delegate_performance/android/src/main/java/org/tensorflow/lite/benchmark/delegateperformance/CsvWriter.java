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

import static org.tensorflow.lite.benchmark.delegateperformance.DelegatePerformanceBenchmark.checkState;

import android.util.Log;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

/** Helper class for writing the final report in CSV format. */
final class CsvWriter implements ReportWriter {
  private static final String TAG = "TfLiteCsvWriter";

  private final String destinationFolderPath;

  private CsvWriter(String destinationFolderPath) {
    this.destinationFolderPath = destinationFolderPath;
  }

  /** Writes the benchmark results into a CSV file. */
  @Override
  public void writeReport(BenchmarkReport report) {
    // Example output file:
    // Model, Metric, DELEGATE_TYPE (PATH), DELEGATE_TYPE (PATH), Change, Status,...
    // model_1, metric_1, 900 , 1000, -10%, PASS, ...
    // model_1, metric_2, ...
    // model_1, delegate_summary,,,, PASS, ...
    // model_1, model_summary, PASS,
    // model_2, ...
    // ...
    // Summary, Summary, PASS
    StringBuilder sb = new StringBuilder();
    sb.append(destinationFolderPath).append("/").append(report.name()).append(".csv");
    String filePath = sb.toString();
    Log.i(TAG, "Generating CSV report to " + filePath);

    List<ModelBenchmarkReportInterface> modelReports = report.modelBenchmarkReports();
    checkState(!modelReports.isEmpty());
    sb = new StringBuilder();
    // Heading row. It is structured as below:
    // Model, Metric, DELEGATE_TYPE (PATH), DELEGATE_TYPE (PATH), Change, Status, ...
    sb.append("Model, Metric");
    for (DelegateMetricsEntry entry : modelReports.get(0).processedDelegateMetrics()) {
      sb.append(", Delegate: ").append(entry.delegateIdentifier());
      if (!entry.isTestTarget()) {
        sb.append(", Change, Status");
      }
    }
    sb.append('\n');
    for (ModelBenchmarkReportInterface modelReport : modelReports) {
      writerModelReport(modelReport, sb);
    }
    sb.append("Summary").append(", Summary,").append(report.result());
    sb.append('\n');
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
        sb.append(modelName).append(",").append(metricName);
        for (DelegateMetricsEntry delegateMetricsEntry : modelReport.processedDelegateMetrics()) {
          MetricsEntry metricEntry = delegateMetricsEntry.metrics().get(metricName);
          sb.append(",").append(metricEntry.value());
          if (!delegateMetricsEntry.isTestTarget()) {
            sb.append(",")
                .append(metricEntry.regression())
                .append(",")
                .append(metricEntry.result());
          }
        }
        sb.append('\n');
      }
    }
    sb.append(modelName).append(",delegate_summary");
    for (DelegateMetricsEntry delegateMetricsEntry : modelReport.processedDelegateMetrics()) {
      sb.append(",");
      if (!delegateMetricsEntry.isTestTarget()) {
        // Position the delegate-level result correctly.
        sb.append(",,").append(delegateMetricsEntry.result());
      }
    }
    sb.append('\n');
    sb.append(modelName).append(",model_summary,").append(modelReport.result());
    sb.append('\n');
  }

  static ReportWriter create(String destinationFolderPath) {
    return new CsvWriter(destinationFolderPath);
  }
}
