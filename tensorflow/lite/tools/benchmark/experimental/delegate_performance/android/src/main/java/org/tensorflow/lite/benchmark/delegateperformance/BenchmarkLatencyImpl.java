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

import static org.tensorflow.lite.benchmark.delegateperformance.DelegatePerformanceBenchmark.checkArgument;
import static org.tensorflow.lite.benchmark.delegateperformance.DelegatePerformanceBenchmark.checkNotNull;
import static org.tensorflow.lite.benchmark.delegateperformance.DelegatePerformanceBenchmark.checkState;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.os.Trace;
import android.util.Log;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import tflite.StableDelegateLoaderSettings;
import tflite.proto.benchmark.DelegatePerformance.LatencyCriteria;
import tflite.proto.benchmark.DelegatePerformance.LatencyResults;

/**
 * Impl class for Delegate Performance Latency Benchmark.
 *
 * <p>It performs latency benchmark tests via TFLite Benchmark Tool based on the input arguments.
 * Please check the test example in
 * tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/README.md.
 *
 * <p>Generates a CSV file under delegate_performance_result/latency folder to describe the
 * benchmark results for each model.
 *
 * <ul>
 *   <li>1. delegate_performance_result/latency/<MODEL_NAME>.csv: the performance of each
 *       acceleration configuration and relative performance differences in percentage values.
 * </ul>
 *
 * <p>Generates a Pass/Fail decision in the below two cases:
 *
 * <ul>
 *   <li>1. the caller provides one TFLite Settings file. The Activity compares the latency
 *       performance of the provided acceleration configuration against the default acceleration
 *       configuation (blank TFLiteSettings). If the regression on the average inference latency
 *       metric is within the corresponding threshold specified in the LatencyCriteria file for all
 *       models, the Activity logs Pass. Otherwise, it logs Fail.
 *   <li>2. the caller provides two TFLite Settings file with stable delegate loader settings. The
 *       activity uses the first one as reference and the other as the test target. If any
 *       regressions on latency metrics are within the thresholds specified in the LatencyCriteria
 *       file, the Activity logs Pass. Otherwise, it logs Fail.
 * </ul>
 *
 * The metrics for generating the Pass/Fail decision:
 *
 * <ul>
 *   <li>1. initialization latency: model loading and interpreter initialization before the first
 *       inference. This metric is only used in the above case #2.
 *   <li>2. averge warmup latency: average time for the warmup inferences before the benchmark run.
 *       The default number of warmup inferences is 1. This metric is only used in the above case
 *       #2.
 *   <li>3. average inference latency: average time for the inferences in the benchmark run.
 * </ul>
 */
public final class BenchmarkLatencyImpl {

  private static final String LATENCY_FOLDER_NAME = "latency";
  private static final String PROTO_FOLDER_NAME = "proto";
  private static final String TAG = "TfLiteLatencyImpl";
  private static final String DEFAULT_LATENCY_CRITERIA_FILENAME = "default_latency_criteria";
  private static final String LATENCY_CRITERIA_FILE_EXT = ".binarypb";
  // Reference entry is the first item in the TfLiteSettingsListEntry list.
  private static final int REFERENCE_ENTRY_INDEX = 0;
  // The test target entry is the second item in the TfLiteSettingsListEntry list.
  private static final int TEST_TARGET_ENTRY_INDEX = 1;

  private final Context context;
  private final String[] tfliteSettingsJsonFiles;
  private final String[] args;
  private LatencyCriteria defaultLatencyCriteria;
  private String resultFolderPath;
  /**
   * {@code true} if the caller provides one delegate to this activity. If the flag is {@code true},
   * the activity generates a PASS/FAIL result in the logs.
   */
  private boolean numberOfInputTfLiteSettingsIsOne;
  /**
   * {@code true} if the caller provides two stable delegates to this activity. If the flag is
   * {@code true}, the activity generates a PASS/FAIL result in the logs.
   */
  private boolean compareTwoStableDelegates;

  public BenchmarkLatencyImpl(Context context, String[] tfliteSettingsJsonFiles, String[] args) {
    this.context = context;
    this.tfliteSettingsJsonFiles = tfliteSettingsJsonFiles;
    if (args == null) {
      // The "--args" extra key was not provided.
      this.args = new String[0];
    } else {
      this.args = args;
    }
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
    numberOfInputTfLiteSettingsIsOne = tfliteSettingsJsonFiles.length == 1;

    try {
      // Creates root result folder.
      resultFolderPath =
          DelegatePerformanceBenchmark.createResultFolder(
              context.getFilesDir(), LATENCY_FOLDER_NAME);
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
    TfLiteSettingsListEntry target = tfliteSettingsList.get(TEST_TARGET_ENTRY_INDEX);
    compareTwoStableDelegates =
        tfliteSettingsList.size() == 2
            && isStableDelegate(tfliteSettingsList.get(REFERENCE_ENTRY_INDEX))
            && isStableDelegate(target);

    boolean passed = true;
    String[] assets;
    try {
      assets = context.getAssets().list(LATENCY_FOLDER_NAME);
    } catch (IOException e) {
      Log.e(TAG, "Failed to list files from assets folder.", e);
      return;
    }
    for (String asset : assets) {
      if (asset.endsWith(".tflite")) {
        passed &= benchmarkModel(asset, tfliteSettingsList, args) == BenchmarkResultType.PASS;
      }
    }
    // TODO(b/250877013): Improve the result reporting.
    if (shouldGeneratePassFailDecision()) {
      Log.i(
          TAG,
          String.format(
              "Latency benchmark result for %s: %s",
              target.filePath(), passed ? BenchmarkResultType.PASS : BenchmarkResultType.FAIL));
    } else {
      Log.i(
          TAG,
          "Skipping the Pass/Fail result generation because the activity receives 2 TFLiteSettings"
              + " JSON files and at least one of the files is not using the stable delegate.");
    }
  }

  /**
   * Benchmarks a model file with the TfLiteSettingsListEntry list.
   *
   * <p>Returns {@code BenchmarkResultType.SKIP} is the latency module shouldn't produce a Pass/Fail
   * result. Otherwise, returns {@code BenchmarkResultType.PASS} if the test target acceleration
   * configuration doesn't breach the thresholds in the or {@code BenchmarkResultType.FAIL} if not.
   * latency criteria file. Returns {@code BenchmarkResultType.UNKONWN} if the benchmark task
   * encounters errors.
   */
  private BenchmarkResultType benchmarkModel(
      String modelFilename, List<TfLiteSettingsListEntry> tfliteSettingsList, String[] args) {
    String modelName = DelegatePerformanceBenchmark.getModelName(modelFilename);
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
        Trace.beginSection("Latency Benchmark");
        LatencyResults results =
            DelegatePerformanceBenchmark.runLatencyBenchmark(
                args,
                tfliteSettingsListEntry,
                modelFileDescriptor.getParcelFileDescriptor().getFd(),
                modelFileDescriptor.getStartOffset(),
                modelFileDescriptor.getLength());
        Trace.endSection();
        tfliteSettingsListEntry.setLatencyResults(results);
      }

      CsvWriter.writeReport(tfliteSettingsList, resultFolderPath + "/" + modelName + ".csv");
      if (!shouldGeneratePassFailDecision()) {
        return BenchmarkResultType.SKIP;
      }
      return checkLatencyCriteria(tfliteSettingsList, tryLoadLatencyCriteria(modelName))
          ? BenchmarkResultType.PASS
          : BenchmarkResultType.FAIL;
    } catch (IOException e) {
      Log.e(TAG, "Failed to open asset file " + LATENCY_FOLDER_NAME + "/" + modelFilename);
    }
    return BenchmarkResultType.UNKONWN;
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

  /**
   * Uses the latency criteria to check the benchmark results.
   *
   * <p>Returns true if the results meet any of the follow conditions:
   *
   * <ul>
   *   <li>1. the caller provides one TFLite Settings file. The average latency time regression is
   *       within the threshold specified in the latency criteria file after comparing the test
   *       target accleration configuration against the default acceleration configuration.
   *   <li>2. the caller provides two TFLite Settings file with stable delegate loader settings. The
   *       method uses the first acceleration configuratin as reference configuration. The
   *       initialization, average warmup and average inference latency regression are within the
   *       thresholds specified in the latency criteria file after comparing the second
   *       configuration against the reference configuration.
   * </ul>
   *
   * TODO(b/250877013): Consider improving the result aggregation logic.
   */
  private boolean checkLatencyCriteria(
      List<TfLiteSettingsListEntry> tfliteSettingsList, LatencyCriteria latencyCriteria) {
    checkState(shouldGeneratePassFailDecision());
    checkNotNull(latencyCriteria);
    // This method checks the latency criteria when the number of entries is two.
    checkArgument(tfliteSettingsList.size() == 2);

    TfLiteSettingsListEntry reference = tfliteSettingsList.get(REFERENCE_ENTRY_INDEX);
    TfLiteSettingsListEntry target = tfliteSettingsList.get(TEST_TARGET_ENTRY_INDEX);
    boolean checkInferenceRegression =
        checkLatencyThreshold(
            reference,
            target,
            "inference_latency_average_us",
            latencyCriteria.getAverageInferenceMaxRegressionPercentageAllowed());
    if (numberOfInputTfLiteSettingsIsOne) {
      // Check for inference latency regression only when the number of input files is one.
      return checkInferenceRegression;
    }
    boolean checkInitializationRegression =
        checkLatencyThreshold(
            reference,
            target,
            "initialization_latency_us",
            latencyCriteria.getInitializationMaxRegressionPercentageAllowed());
    boolean checkWarmupRegression =
        checkLatencyThreshold(
            reference,
            target,
            "warmup_latency_average_us",
            latencyCriteria.getAverageWarmUpMaxRegressionPercentageAllowed());
    return checkInferenceRegression && checkInitializationRegression && checkWarmupRegression;
  }

  /**
   * Currently this Activity generates a Pass/Fail result when it receives two stable delegate
   * acceleration configurations or one acceleration configuration. Because the result is generated
   * by comparing the same metrics between 2 delegates. So the comparison is fair if the two
   * delegates are with the same delegate type or it is comparing with the default delegate, which
   * is used when no acceleration configuration is provided.
   *
   * <p>TODO(b/250877013): Consider improving the I/O of this activity.
   *
   * <p>Returns true if the latency module receives two stable delegate acceleration configurations
   * or one acceleration configuration. Otherwise, returns false.
   */
  private boolean shouldGeneratePassFailDecision() {
    return compareTwoStableDelegates || numberOfInputTfLiteSettingsIsOne;
  }

  /**
   * Returns true if the {@code TfLiteSettingsListEntry} refers to a stable delegate. Otherwise,
   * returns false.
   *
   * <p>This Activity generates a Pass/Fail result if both of the two input TFLiteSettings JSON
   * files refer to stable delegates. This method a helper function to know if a {@code
   * TfLiteSettingsListEntry} refers to a stable delegate.
   */
  private boolean isStableDelegate(TfLiteSettingsListEntry entry) {
    StableDelegateLoaderSettings settings = entry.tfliteSettings().stableDelegateLoaderSettings();
    return settings != null
        && settings.delegatePath() != null
        && !settings.delegatePath().isEmpty();
  }

  /** Checks if the regression metric is within the thresholds provided. */
  private boolean checkLatencyThreshold(
      TfLiteSettingsListEntry reference,
      TfLiteSettingsListEntry target,
      String metricName,
      float percentageThreshold) {
    if (reference.metrics().containsKey(metricName) && target.metrics().containsKey(metricName)) {
      float referenceMetricValue = reference.metrics().get(metricName);
      float targetMetricValue = target.metrics().get(metricName);
      if (referenceMetricValue == 0) {
        return targetMetricValue == referenceMetricValue;
      }
      return (targetMetricValue - referenceMetricValue) / referenceMetricValue
          <= percentageThreshold / 100f;
    }
    return false;
  }
}
