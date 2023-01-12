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
import android.os.Trace;
import android.util.Log;
import java.io.IOException;
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
 * <p>TODO(b/250877013): Consider improving the app's I/O interfaces.
 *
 * <p>Generates a Pass/Fail result. The test is a Pass if the target acceleration configuration (the
 * second configuration if more than one TFLiteSettings JSON files are provided) passes the embedded
 * metric thresholds in all models.
 *
 * <p>Generates a CSV file for each model to describe the benchmark results under
 * delegate_performance_result/accuracy folder in the app files directory.
 *
 * <ul>
 *   <li>1. delegate_performance_result/accuracy/<MODEL_NAME>.csv: the performance of each
 *       acceleration configuration and relative performance differences in percentage values.
 * </ul>
 */
public class BenchmarkAccuracyImpl {

  private static final String TAG = "TfLiteAccuracyImpl";
  private static final String ACCURACY_FOLDER_NAME = "accuracy";
  // The test target entry is the second item in the TfLiteSettingsListEntry list.
  private static final int TEST_TARGET_ENTRY_INDEX = 1;

  private final Context context;
  private final String[] tfliteSettingsJsonFiles;
  private String resultFolderPath;

  public BenchmarkAccuracyImpl(Context context, String[] tfliteSettingsJsonFiles) {
    this.context = context;
    this.tfliteSettingsJsonFiles = tfliteSettingsJsonFiles;
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
      resultFolderPath =
          DelegatePerformanceBenchmark.createResultFolder(
              context.getFilesDir(), ACCURACY_FOLDER_NAME);
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
    boolean passed = true;
    TfLiteSettingsListEntry targetEntry = tfliteSettingsList.get(TEST_TARGET_ENTRY_INDEX);
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
        Log.e(TAG, "Failed to create result folder for " + modelName, e);
        passed = false;
        break;
      }
      try (AssetFileDescriptor modelFileDescriptor =
          context.getAssets().openFd(ACCURACY_FOLDER_NAME + "/" + asset)) {
        for (TfLiteSettingsListEntry tfliteSettingsListEntry : tfliteSettingsList) {
          Trace.beginSection("Accuracy Benchmark");
          BenchmarkEvent benchmarkEvent =
              DelegatePerformanceBenchmark.runAccuracyBenchmark(
                  tfliteSettingsListEntry,
                  modelFileDescriptor.getParcelFileDescriptor().getFd(),
                  modelFileDescriptor.getStartOffset(),
                  modelFileDescriptor.getLength(),
                  modelResultPath);
          Trace.endSection();

          tfliteSettingsListEntry.setAccuracyResults(benchmarkEvent);
        }

        passed &= targetEntry.metrics().containsKey("ok") && targetEntry.metrics().get("ok") > 0;
        CsvWriter.writeReport(
            tfliteSettingsList, String.format("%s/%s.csv", resultFolderPath, modelName));
      } catch (IOException e) {
        Log.e(TAG, "Failed to open assets file " + asset, e);
        passed = false;
        break;
      }
    }
    Log.i(
        TAG,
        String.format(
            "Accuracy benchmark result for %s: %s.",
            targetEntry.filePath(), passed ? "Pass" : "Fail"));
  }
}
