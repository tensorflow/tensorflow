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

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import tflite.proto.benchmark.DelegatePerformance.LatencyCriteria;
import tflite.proto.benchmark.DelegatePerformance.LatencyResults;

/**
 * Impl class for Delegate Performance Latency Benchmark.
 *
 * <p>It performs latency benchmark tests via TFLite Benchmark Tool based on the input arguments.
 * Please check the test example in
 * tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/README.md.
 *
 * <p>Generates below list of files under delegate_performance_result/latency folder to describe the
 * benchmark results.
 *
 * <ul>
 *   <li>1. delegate_performance_result/latency/report.csv: the performance of each acceleration
 *       configuration and relative performance differences as percentages in CSV.
 *   <li>2. delegate_performance_result/latency/report.json: detailed performance results. The file
 *       contains the metric-level, delegate-level and model-level results, the latency criteria and
 *       the raw metric outputs from the native layer in JSON.
 *   <li>3. delegate_performance_result/latency/report.html: the performance of each acceleration
 *       configuration and relative performance differences as percentages in HTML.
 * </ul>
 *
 * The metrics for generating the Pass/Pass with Warning/Fail decision:
 *
 * <ul>
 *   <li>1. startup overhead latency: it is equal to (initialization time + average warmup latency -
 *       average inference time).
 *   <li>2. average inference latency: average time for the inferences in the benchmark run.
 * </ul>
 */
public final class BenchmarkLatencyImpl {

  private static final String LATENCY_FOLDER_NAME = "latency";
  private static final String PROTO_FOLDER_NAME = "proto";
  private static final String TAG = "TfLiteLatencyImpl";
  private static final String DEFAULT_LATENCY_CRITERIA_FILENAME = "default_latency_criteria";
  private static final String LATENCY_CRITERIA_FILE_EXT = ".binarypb";

  private final Context context;
  private final String[] tfliteSettingsJsonFiles;
  private final String[] args;
  private final BenchmarkReport report;
  private LatencyCriteria defaultLatencyCriteria;

  public BenchmarkLatencyImpl(Context context, String[] tfliteSettingsJsonFiles, String[] args) {
    this.context = context;
    this.tfliteSettingsJsonFiles = tfliteSettingsJsonFiles;
    if (args == null) {
      // The "--args" extra key was not provided.
      this.args = new String[0];
    } else {
      this.args = args;
    }
    this.report = BenchmarkReport.create();
  }

  /**
   * Initializes the test environment. Creates the result folder and loads the default latency
   * criteria file.
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
              context.getFilesDir(), LATENCY_FOLDER_NAME);
      report.addWriter(JsonWriter.create(resultFolderPath));
      report.addWriter(CsvWriter.create(resultFolderPath));
      report.addWriter(HtmlWriter.create(resultFolderPath));
    } catch (IOException e) {
      Log.e(
          TAG, "Failed to create result folder " + LATENCY_FOLDER_NAME + " in files directory.", e);
      return false;
    }

    try {
      // Loads default latency criteria.
      defaultLatencyCriteria = loadLatencyCriteria(DEFAULT_LATENCY_CRITERIA_FILENAME);
    } catch (IOException e) {
      Log.e(TAG, "Failed to load default latency criteria " + DEFAULT_LATENCY_CRITERIA_FILENAME, e);
      return false;
    }
    return true;
  }

  /** Benchmarks the embedded model files with the input TFLiteSettings JSON files. */
  public void benchmark() {
    List<TfLiteSettingsListEntry> tfliteSettingsList =
        DelegatePerformanceBenchmark.loadTfLiteSettingsList(tfliteSettingsJsonFiles);
    if (tfliteSettingsList.size() < 2) {
      Log.e(TAG, "Failed to load the TFLiteSettings JSON file.");
      return;
    }

    String[] assets;
    try {
      assets = context.getAssets().list(LATENCY_FOLDER_NAME);
    } catch (IOException e) {
      Log.e(TAG, "Failed to list files from assets folder.", e);
      return;
    }
    for (String asset : assets) {
      if (asset.endsWith(".tflite")) {
        report.addModelBenchmarkReport(benchmarkModel(asset, tfliteSettingsList, args));
      }
    }
    // Computes the aggregated results and export the report to local files.
    report.export();
    TfLiteSettingsListEntry testTarget = tfliteSettingsList.get(tfliteSettingsList.size() - 1);
    checkState(testTarget.isTestTarget());
    Log.i(
        TAG,
        String.format(
            "Latency benchmark result for %s: %s", testTarget.filePath(), report.result()));
  }

  /**
   * Benchmarks a model file with the TfLiteSettingsListEntry list.
   *
   * <p>Returns {@link ModelBenchmarkReportInterface}, which is the model level benchmark report
   * that has the delegate information and the raw metric results from the native layer.
   */
  private ModelBenchmarkReportInterface benchmarkModel(
      String modelFilename, List<TfLiteSettingsListEntry> tfliteSettingsList, String[] args) {
    String modelName = DelegatePerformanceBenchmark.getModelName(modelFilename);
    LatencyCriteria latencyCriteria = tryLoadLatencyCriteria(modelName);
    ModelBenchmarkReport<LatencyResults> report =
        LatencyBenchmarkReport.create(modelName, latencyCriteria);
    try (AssetFileDescriptor modelFileDescriptor =
        context.getAssets().openFd(LATENCY_FOLDER_NAME + "/" + modelFilename)) {
      for (TfLiteSettingsListEntry tfliteSettingsListEntry : tfliteSettingsList) {
        Log.i(
            TAG,
            "Running latency benchmark with model: "
                + modelName
                + ", settings: "
                + tfliteSettingsListEntry.filePath()
                + ", args: "
                + Arrays.toString(args));
        LatencyResults results =
            DelegatePerformanceBenchmark.runLatencyBenchmark(
                args,
                tfliteSettingsListEntry,
                modelFileDescriptor.getParcelFileDescriptor().getFd(),
                modelFileDescriptor.getStartOffset(),
                modelFileDescriptor.getLength());
        report.addResults(results, tfliteSettingsListEntry);
      }
    } catch (IOException e) {
      Log.e(TAG, "Failed to open asset file " + LATENCY_FOLDER_NAME + "/" + modelFilename);
    }
    return report;
  }

  /**
   * Tries to load the model-specific latency criteria file by the model name.
   *
   * <p>Returns the latency criteria for the specific model if the loading was successful.
   * Otherwise, returns the default latency criteria.
   */
  private LatencyCriteria tryLoadLatencyCriteria(String fileBasename) {
    try {
      return loadLatencyCriteria(fileBasename);
    } catch (IOException e) {
      Log.w(
          TAG,
          "Failed to load the latency criteria of "
              + fileBasename
              + ". Fallback to the default latency criteria.");
    }
    return defaultLatencyCriteria;
  }

  /** Loads the latency criteria file from Assets. */
  private LatencyCriteria loadLatencyCriteria(String fileBasename) throws IOException {
    String latencyCriteriaFileAssetPath =
        PROTO_FOLDER_NAME + "/" + fileBasename + LATENCY_CRITERIA_FILE_EXT;
    InputStream latencyCriteriaFile = context.getAssets().open(latencyCriteriaFileAssetPath);
    return LatencyCriteria.parseFrom(latencyCriteriaFile);
  }
}
