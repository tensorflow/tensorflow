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
import java.io.File;
import java.io.IOException;

/** Helper class for running delegate performance benchmark. */
class DelegatePerformanceBenchmark {
  private static final String DELEGATE_PERFORMANCE_RESULT_FOLDER = "delegate_performance_result";
  private static final String TAG = "tflite_DelegatePerformanceBenchmark";

  static {
    System.loadLibrary("tensorflowlite_delegate_performance_benchmark");
  }

  public static String createResultFolder(File filesDir, String resultFolder) throws IOException {
    File resultDir = new File(filesDir, DELEGATE_PERFORMANCE_RESULT_FOLDER + "/" + resultFolder);
    String resultPath = resultDir.getAbsolutePath();
    if (resultDir.exists() || resultDir.mkdirs()) {
      Log.i(TAG, "Logging the result to " + resultPath);
      return resultPath;
    }
    throw new IOException("Failed to create directory for " + resultPath);
  }

  public static void runLatencyBenchmark(String[] args, String resultPath) {
    latencyBenchmarkNativeRun(args, resultPath);
  }

  public static int runAccuracyBenchmark(
      String[] args, int modelFd, long modelOffset, long modelSize, String resultPath) {
    return accuracyBenchmarkNativeRun(args, modelFd, modelOffset, modelSize, resultPath);
  }

  private static native void latencyBenchmarkNativeRun(String[] args, String resultPath);

  private static native int accuracyBenchmarkNativeRun(
      String[] args, int modelFd, long modelOffset, long modelSize, String resultPath);
}
