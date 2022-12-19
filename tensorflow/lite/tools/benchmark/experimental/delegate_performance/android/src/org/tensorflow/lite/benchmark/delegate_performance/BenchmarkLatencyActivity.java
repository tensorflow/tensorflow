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

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.os.Trace;
import android.util.Log;
import com.google.protos.tflite.proto.benchmark.DelegatePerformance.LatencyCriteria;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import tflite.StableDelegateLoaderSettings;

/**
 * {@link Activity} class for Delegate Performance Latency Benchmark.
 *
 * <p>This Activity receives test arguments via a command line specified in an intent extra. It
 * performs latency benchmark tests via TFLite Benchmark Tool based on the input arguments. Please
 * check the test example in
 * tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/README.md.
 *
 * <p>Generates a CSV file to describe the benchmark results under
 * delegate_performance_result/latency folder in the app files directory.
 *
 * <ul>
 *   <li>1. delegate_performance_result/latency/model.csv: the performance of each acceleration
 *       configuration and relative performance differences in percentage values.
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
public class BenchmarkLatencyActivity extends Activity {

  private static final String TAG = "tflite_BenchmarkLatencyActivity";
  private static final String LATENCY_RESULT_FOLDER = "latency";
  private static final String TFLITE_SETTINGS_FILES_INTENT_KEY_0 = "--tflite_settings_files";
  private static final String ARGS_INTENT_KEY_0 = "--args";
  // Reference entry is the first item in the TfLiteSettingsListEntry list.
  private static final int REFERENCE_ENTRY_INDEX = 0;
  // The test target entry is the second item in the TfLiteSettingsListEntry list.
  private static final int TEST_TARGET_ENTRY_INDEX = 1;

  @Override
  public void onCreate(Bundle savedInstanceState) {
    Log.i(TAG, "Create benchmark latency activity.");
    super.onCreate(savedInstanceState);

    Intent intent = getIntent();
    Bundle bundle = intent.getExtras();
    String[] tfliteSettingsJsonFiles = bundle.getStringArray(TFLITE_SETTINGS_FILES_INTENT_KEY_0);
    if (tfliteSettingsJsonFiles.length == 0) {
      Log.e(TAG, "No TFLiteSettings file is provided.");
      finish();
      return;
    }
    String[] args = bundle.getStringArray(ARGS_INTENT_KEY_0);

    String resultPath;
    try {
      resultPath =
          DelegatePerformanceBenchmark.createResultFolder(
              getApplicationContext().getFilesDir(), LATENCY_RESULT_FOLDER);
    } catch (IOException e) {
      Log.e(
          TAG,
          "Failed to create result folder " + LATENCY_RESULT_FOLDER + " in files directory.",
          e);
      finish();
      return;
    }

    // TODO(b/250877013): Embed models into the benchmark latency activity. Find the corresponding
    // latency criteria file.
    LatencyCriteria latencyCriteria;
    try {
      InputStream latencyCriteriaFile = getAssets().open("proto/default_latency_criteria.binarypb");
      latencyCriteria = LatencyCriteria.parseFrom(latencyCriteriaFile);
    } catch (IOException e) {
      Log.e(TAG, "Failed to open the latency criteria file", e);
      finish();
      return;
    }

    List<TfLiteSettingsListEntry> tfliteSettingsList =
        DelegatePerformanceBenchmark.loadTfLiteSettingsList(tfliteSettingsJsonFiles);
    if (tfliteSettingsList.size() < 2) {
      Log.e(TAG, "Failed to load the TFLiteSettings JSON file.");
      finish();
      return;
    }

    for (TfLiteSettingsListEntry tfliteSettingsListEntry : tfliteSettingsList) {
      Trace.beginSection("Latency Benchmark");
      tfliteSettingsListEntry.setLatencyResults(
          DelegatePerformanceBenchmark.runLatencyBenchmark(args, tfliteSettingsListEntry));
      Trace.endSection();
    }

    // TODO(b/250877013): Replace model name.
    CsvWriter.writeReport(tfliteSettingsList, resultPath + "/model.csv");
    checkLatencyCriteria(tfliteSettingsJsonFiles.length, tfliteSettingsList, latencyCriteria);
    finish();
  }

  private void checkLatencyCriteria(
      int tfliteSettingsJsonFilesCount,
      List<TfLiteSettingsListEntry> tfliteSettingsList,
      LatencyCriteria latencyCriteria) {
    if (tfliteSettingsList.size() != 2) {
      // The Activity compares the first two entries to generate a Pass/Fail result.
      Log.i(
          TAG,
          "Skipping the Pass/Fail result generation because the number of candidate TFLiteSettings"
              + " is not equal to 2. It is "
              + tfliteSettingsList.size());
      return;
    }
    TfLiteSettingsListEntry reference = tfliteSettingsList.get(REFERENCE_ENTRY_INDEX);
    TfLiteSettingsListEntry target = tfliteSettingsList.get(TEST_TARGET_ENTRY_INDEX);
    boolean compareTwoStableDelegates = isStableDelegate(reference) && isStableDelegate(target);
    if (!compareTwoStableDelegates && tfliteSettingsJsonFilesCount != 1) {
      // The Activity generates a Pass/Fail result when it receives two stable delegate acceleration
      // configurations or one acceleration configuration.
      Log.i(
          TAG,
          "Skip the Pass/Fail result generation because the actvity receives 2 TFLiteSettings JSON"
              + " files and at least one of the files is not using the stable delegate.");
      return;
    }
    boolean checkInferenceRegression =
        checkLatencyThreshold(
            reference,
            target,
            "inference_latency_average_us",
            latencyCriteria.AVERAGE_INFERENCE_MAX_REGRESSION_PERCENTAGE_ALLOWED_FIELD_NUMBER);
    boolean checkInitializationRegression =
        checkLatencyThreshold(
            reference,
            target,
            "initialization_latency_us",
            latencyCriteria.INITIALIZATION_MAX_REGRESSION_PERCENTAGE_ALLOWED_FIELD_NUMBER);
    boolean checkWarmupRegression =
        checkLatencyThreshold(
            reference,
            target,
            "warmup_latency_average_us",
            latencyCriteria.AVERAGE_WARM_UP_MAX_REGRESSION_PERCENTAGE_ALLOWED_FIELD_NUMBER);
    String result = "FAIL";
    if (compareTwoStableDelegates
        && checkInferenceRegression
        && checkInitializationRegression
        && checkWarmupRegression) {
      result = "PASS";
    } else if (tfliteSettingsJsonFilesCount == 1 && checkInferenceRegression) {
      result = "PASS";
    }
    Log.i(TAG, "Latency benchmark result for " + target.filePath() + ": " + result);
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

  private boolean checkLatencyThreshold(
      TfLiteSettingsListEntry reference,
      TfLiteSettingsListEntry target,
      String metricName,
      int percentageThreshold) {
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
