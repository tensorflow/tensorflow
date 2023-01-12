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
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** Helper class for writing the final report. */
final class CsvWriter {
  private static final String TAG = "TfLiteCsvWriter";

  /**
   * Writes the benchmark results into a CSV file.
   *
   * <p>Example output file:
   *
   * <p>| Metric | DELEGATE_TYPE_1 (PATH_1) | DELEGATE_TYPE_2 (PATH_2) | % | ...
   *
   * <p>| METRIC_1 | 1000 | 1200 | 20% | ...
   *
   * <p>...
   */
  public static void writeReport(
      List<TfLiteSettingsListEntry> tfliteSettingsList, String filePath) {
    if (tfliteSettingsList.isEmpty()) {
      Log.e(TAG, "Invalid input to generate a CSV report.");
      return;
    }

    Log.i(TAG, "Generating CSV report to " + filePath);
    TfLiteSettingsListEntry reference = tfliteSettingsList.get(0);
    try (PrintWriter writer = new PrintWriter(filePath)) {
      StringBuilder sb = new StringBuilder();
      // Heading row. It is structured as below:
      // Metric, <REFERENCE_DELEGATE> (<PATH>), <CANDIDATE_DELEGATE> (<PATH>), %,...
      sb.append("Metric,")
          .append(reference.tfliteSettings().delegate())
          .append(" (")
          .append(reference.filePath())
          .append(")");
      for (int i = 1; i < tfliteSettingsList.size(); i++) {
        TfLiteSettingsListEntry entry = tfliteSettingsList.get(i);
        sb.append(",")
            .append(entry.tfliteSettings().delegate())
            .append(" (")
            .append(entry.filePath())
            .append("),%");
      }
      sb.append('\n');

      // Metric rows.
      for (Map.Entry<String, Float> referenceEntry : reference.metrics().entrySet()) {
        String metricName = referenceEntry.getKey();
        float referenceValue = referenceEntry.getValue();
        sb.append(metricName).append(",").append(referenceValue);
        for (int i = 1; i < tfliteSettingsList.size(); i++) {
          sb.append(compareValues(metricName, referenceValue, tfliteSettingsList.get(i)));
        }
        sb.append('\n');
      }

      writer.write(sb.toString());
    } catch (IOException e) {
      Log.e(TAG, "Failed to open report file " + filePath);
    }
  }

  private static String compareValues(
      String metricName, float referenceValue, TfLiteSettingsListEntry entry) {
    StringBuilder sb = new StringBuilder();
    sb.append(",N/A,N/A");
    if (entry.metrics().containsKey(metricName)) {
      float value = entry.metrics().get(metricName);
      sb.setLength(0);
      sb.append(",").append(value);
      if (value == referenceValue) {
        sb.append(",0%");
      } else if (referenceValue == 0) {
        sb.append(",N/A");
      } else {
        sb.append(",").append(toPercentage((value - referenceValue) / referenceValue));
      }
    }
    return sb.toString();
  }

  private static String toPercentage(float n) {
    return String.format(Locale.ENGLISH, "%.1f", n * 100) + "%";
  }
}
