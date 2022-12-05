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
import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.os.Trace;
import android.util.Log;
import java.io.IOException;
import java.util.Arrays;

/**
 * {@link Activity} class for Delegate Performance Accuracy Benchmark.
 *
 * <p>This Activity receives test arguments via a command line specified in an intent extra. It
 * performs accuracy benchmark tests via TFLite MiniBenchmark based on the input arguments. Please
 * check the test example in
 * https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/README.md.
 * Current version doesn't write files to app file directory. The results can only be checked via
 * logcat logs.
 *
 * <p>TODO(b/250877013): add performance thresholding into the app. The activity will produce a
 * PASS/FAIL result based on the thresholds.
 */
public class BenchmarkAccuracyActivity extends Activity {

  private static final String TAG = "tflite_BenchmarkAccuracyActivity";
  private static final String ACCURACY_RESULT_FOLDER = "accuracy";
  private static final String ARGS_INTENT_KEY_0 = "--args";

  @Override
  public void onCreate(Bundle savedInstanceState) {
    Log.i(TAG, "Create benchmark accuracy activity.");
    super.onCreate(savedInstanceState);

    Intent intent = getIntent();
    Bundle bundle = intent.getExtras();
    String[] args = bundle.getStringArray(ARGS_INTENT_KEY_0);

    try {
      String resultPath =
          DelegatePerformanceBenchmark.createResultFolder(
              getApplicationContext().getFilesDir(), ACCURACY_RESULT_FOLDER);

      Log.i(TAG, "Running accuracy benchmark with args: " + Arrays.toString(args));
      benchmarkAccuracy(args, resultPath);
    } catch (IOException e) {
      Log.e(TAG, "Failed to create result folder", e);
    }

    finish();
  }

  private void benchmarkAccuracy(String[] args, String resultPath) {
    String[] assets;
    try {
      assets = getAssets().list("");
    } catch (IOException e) {
      Log.e(TAG, "Failed to list files from assets folder.", e);
      return;
    }
    for (String asset : assets) {
      // TODO(b/252976498): Move the embedded models to a specific folder under Assets.
      if (!asset.endsWith(".tflite")) {
        continue;
      }
      try (AssetFileDescriptor modelFileDescriptor = getAssets().openFd(asset)) {
        Trace.beginSection("Accuracy Benchmark");
        int status =
            DelegatePerformanceBenchmark.runAccuracyBenchmark(
                args,
                modelFileDescriptor.getParcelFileDescriptor().getFd(),
                modelFileDescriptor.getStartOffset(),
                modelFileDescriptor.getLength(),
                resultPath);
        Trace.endSection();
        Log.i(TAG, "Accuracy benchmark of " + asset + " finished with status: " + status);
      } catch (IOException e) {
        Log.e(TAG, "Failed to open assets file " + asset, e);
      }
    }
  }
}
