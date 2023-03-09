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

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import tflite.BenchmarkEvent;

/**
 * Impl class for Delegate Performance Accuracy Benchmark.
 *
 * <p>It performs accuracy benchmark tests via TFLite MiniBenchmark based on the input arguments.
 * Please check the test example in
 * tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/README.md.
 *
 * <p>Generates a PASS/PASS_WITH_WARNING/FAIL result.
 *
 * <ul>
 *   <li>PASS: The test target delegate passed the embedded metric thresholds in all models.
 *   <li>PASS_WITH_WARNING: Both the test target delegate and the reference delegates breached the
 *       embedded metric thresholds.
 *   <li>FAIL: The test target delegate failed at least 1 embedded metric threshold in the models,
 *       and at least 1 reference delegate passed the embedded metric thresholds in all models.
 * </ul>
 *
 * <p>Generates below list of files to describe the benchmark results under
 * delegate_performance_result/accuracy folder in the app files directory.
 *
 * <ul>
 *   <li>1. delegate_performance_result/accuracy/report.csv: the performance of each acceleration
 *       configuration and relative performance differences as percentages in CSV.
 *   <li>2. delegate_performance_result/accuracy/report.json: detailed performance results. The file
 *       contains the metric-level, delegate-level and model-level results and the raw metric
 *       outputs from the native layer in JSON.
 *   <li>3. delegate_performance_result/accuracy/report.html: the performance of each acceleration
 *       configuration and relative performance differences as percentages in HTML.
 * </ul>
 */
public class BenchmarkAccuracyImpl {

  private static final String TAG = "TfLiteAccuracyImpl";
  private static final String ACCURACY_FOLDER_NAME = "accuracy";

  private final Context context;
  private final String[] tfliteSettingsJsonFiles;
  private final BenchmarkReport report;

  public BenchmarkAccuracyImpl(Context context, String[] tfliteSettingsJsonFiles) {
    this.context = context;
    this.tfliteSettingsJsonFiles = tfliteSettingsJsonFiles;
    this.report = BenchmarkReport.create();
  }

  /**
   * Initializes the test environment. Checks the validity of input arguments and creates the result
   * folder.
   *
   * <p>Returns {@code true} if the initialization was successful. Otherwise, returns {@code false}.
   */
  public boolean initialize() {
    if (tfliteSettingsJsonFiles == null || tfliteSettingsJsonFiles.length == 0) {
      Log.e(TAG, "No TFLiteSettings file provided.");
      return false;
    }

    try {
      // Creates root result folder.
      String resultFolderPath =
          DelegatePerformanceBenchmark.createResultFolder(
              context.getFilesDir(), ACCURACY_FOLDER_NAME);
      report.addWriter(JsonWriter.create(resultFolderPath));
      report.addWriter(CsvWriter.create(resultFolderPath));
      report.addWriter(HtmlWriter.create(resultFolderPath));
    } catch (IOException e) {
      Log.e(TAG, "Failed to create result folder", e);
      return false;
    }
    return true;
  }

  public void benchmark() {
    Log.i(
        TAG,
        "Running accuracy benchmark with TFLiteSettings JSON files: "
            + Arrays.toString(tfliteSettingsJsonFiles));
    List<TfLiteSettingsListEntry> tfliteSettingsList =
        DelegatePerformanceBenchmark.loadTfLiteSettingsList(tfliteSettingsJsonFiles);
    if (tfliteSettingsList.size() < 2) {
      Log.e(TAG, "Failed to load the TFLiteSettings JSON file.");
      return;
    }
    String[] assets;
    try {
      assets = context.getAssets().list(ACCURACY_FOLDER_NAME);
    } catch (IOException e) {
      Log.e(TAG, "Failed to list files from assets folder.", e);
      return;
    }
    for (String asset : assets) {
      if (!asset.endsWith(".tflite")) {
        Log.i(TAG, asset + " is not a model file. Skipping.");
        continue;
      }
      String modelResultPath;
      String modelName = DelegatePerformanceBenchmark.getModelName(asset);
      try {
        modelResultPath =
            DelegatePerformanceBenchmark.createResultFolder(
                context.getFilesDir(), ACCURACY_FOLDER_NAME + "/" + modelName);
      } catch (IOException e) {
        Log.e(TAG, "Failed to create result folder for " + modelName + ". Exiting application.", e);
        return;
      }
      try (AssetFileDescriptor modelFileDescriptor =
          context.getAssets().openFd(ACCURACY_FOLDER_NAME + "/" + asset)) {
        List<RawDelegateMetricsEntry> rawDelegateMetricsEntries = new ArrayList<>();
        for (TfLiteSettingsListEntry tfliteSettingsListEntry : tfliteSettingsList) {
          BenchmarkEvent benchmarkEvent =
              DelegatePerformanceBenchmark.runAccuracyBenchmark(
                  tfliteSettingsListEntry,
                  modelFileDescriptor.getParcelFileDescriptor().getFd(),
                  modelFileDescriptor.getStartOffset(),
                  modelFileDescriptor.getLength(),
                  modelResultPath);

          rawDelegateMetricsEntries.add(
              AccuracyBenchmarkReport.parseResults(benchmarkEvent, tfliteSettingsListEntry));
        }
        report.addModelBenchmarkReport(
            AccuracyBenchmarkReport.create(modelName, rawDelegateMetricsEntries));
      } catch (IOException e) {
        Log.e(TAG, "Failed to open assets file " + asset, e);
        return;
      }
    }
    // Computes the aggregated results and export the report to local files.
    report.export();
    TfLiteSettingsListEntry testTarget = tfliteSettingsList.get(tfliteSettingsList.size() - 1);
    Log.i(
        TAG,
        String.format(
            "Accuracy benchmark result for %s: %s.", testTarget.filePath(), report.result()));
  }
}
