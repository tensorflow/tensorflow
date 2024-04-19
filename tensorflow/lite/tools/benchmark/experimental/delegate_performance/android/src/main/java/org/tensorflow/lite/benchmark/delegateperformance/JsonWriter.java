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

import android.util.Log;
import java.io.IOException;
import java.io.PrintWriter;
import org.json.JSONException;

/** Helper class for writing the final report in JSON format. */
final class JsonWriter implements ReportWriter {
  private static final String TAG = "TfLiteJsonWriter";

  private final String destinationFolderPath;

  private JsonWriter(String destinationFolderPath) {
    this.destinationFolderPath = destinationFolderPath;
  }

  // Writes the benchmark results into a JSON file.
  // Example output file:
  // {
  //    "reports": [
  //       {
  //          "model": "model_name",
  //          "metrics": [...],
  //          "raw_metrics": [... ],
  //          "max_regression_percentage_allowed": { ... },
  //          "result": "PASS"
  //       }
  //       , ...
  //    ],
  //    "name": "report",
  //    "result": "PASS"
  // }
  @Override
  public void writeReport(BenchmarkReport report) {
    StringBuilder sb = new StringBuilder();
    sb.append(destinationFolderPath).append("/").append(report.name()).append(".json");
    String filePath = sb.toString();
    try (PrintWriter writer = new PrintWriter(filePath, "UTF-8")) {
      writer.println(report.toJsonObject().toString(/* indentFactor= */ 4));
      Log.i(TAG, "Generated report " + filePath + ".");
    } catch (IOException e) {
      Log.e(TAG, "Failed to open report file " + filePath + "." + e);
    } catch (JSONException e) {
      Log.e(TAG, "Failed to convert the benchmark report to JSON." + e);
    }
  }

  static ReportWriter create(String destinationFolderPath) {
    return new JsonWriter(destinationFolderPath);
  }
}
