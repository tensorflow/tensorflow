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
import java.io.IOException;
import java.util.Arrays;

/**
 * {@link Activity} class for Delegate Performance Latency Benchmark.
 *
 * <p>This Activity receives test arguments via a command line specified in an intent extra. It
 * performs latency benchmark tests via TFLite Benchmark Tool based on the input arguments. Please
 * check the test example in
 * https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/README.md.
 * Generates a JSON file and a CSV file to describe the benchmark results under
 * delegate_performance_result/latency folder in the app files directory.
 *
 * <ul>
 *   <li>1. delegate_performance_result/latency/report.json: the success status of the benchmark
 *       run.
 *   <li>2. delegate_performance_result/latency/benchmark_result.csv: the performance results of the
 *       benchmark run.
 * </ul>
 *
 * <p>TODO(b/250877013): add performance thresholding into the app. The activity will produce a
 * PASS/FAIL result based on the thresholds.
 */
public class BenchmarkLatencyActivity extends Activity {

  private static final String TAG = "tflite_BenchmarkLatencyActivity";
  private static final String LATENCY_RESULT_FOLDER = "latency";

  private static final String ARGS_INTENT_KEY_0 = "--args";

  @Override
  public void onCreate(Bundle savedInstanceState) {
    Log.i(TAG, "Create benchmark latency activity.");
    super.onCreate(savedInstanceState);

    Intent intent = getIntent();
    Bundle bundle = intent.getExtras();
    String[] args = bundle.getStringArray(ARGS_INTENT_KEY_0);

    try {
      String resultPath =
          DelegatePerformanceBenchmark.createResultFolder(
              getApplicationContext().getFilesDir(), LATENCY_RESULT_FOLDER);

      Log.i(TAG, "Running latency benchmark with args: " + Arrays.toString(args));
      Trace.beginSection("Latency Benchmark");
      DelegatePerformanceBenchmark.runLatencyBenchmark(args, resultPath);
      Trace.endSection();
    } catch (IOException e) {
      Log.e(TAG, "Failed to create result folder", e);
    }

    finish();
  }
}
