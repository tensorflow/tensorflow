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
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.protos.tflite.proto.benchmark.DelegatePerformance.BenchmarkEventType;
import com.google.protos.tflite.proto.benchmark.DelegatePerformance.LatencyResults;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import tflite.BenchmarkEvent;
import tflite.TFLiteSettings;

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

  /**
   * Returns a {@code LatencyResults} by parsing the outcome from a TFLite Benchmark Tool execution.
   * If it fails to parse the outcome, this method returns a {@code LatencyResults} with an error
   * event type.
   */
  public static LatencyResults runLatencyBenchmark(
      String[] args, TfLiteSettingsListEntry tfliteSettingslistEntry) {
    byte[] tfliteSettingsByteArray =
        new byte[tfliteSettingslistEntry.tfliteSettings().getByteBuffer().remaining()];
    tfliteSettingslistEntry.tfliteSettings().getByteBuffer().get(tfliteSettingsByteArray);
    byte[] latencyResultsByteArray =
        latencyBenchmarkNativeRun(
            args, tfliteSettingsByteArray, tfliteSettingslistEntry.filePath());
    try {
      return LatencyResults.parseFrom(latencyResultsByteArray);
    } catch (IOException e) {
      Log.i(TAG, "Failed to parse the results running " + tfliteSettingslistEntry.filePath());
      return LatencyResults.newBuilder()
          .setEventType(BenchmarkEventType.BENCHMARK_EVENT_TYPE_ERROR)
          .build();
    }
  }

  /** Returns a {@code BenchmarkEvent} by parsing the outcome from a MiniBenchmark execution. */
  public static BenchmarkEvent runAccuracyBenchmark(
      TfLiteSettingsListEntry tfliteSettingslistEntry,
      int modelFd,
      long modelOffset,
      long modelSize,
      String resultPath) {
    byte[] tfliteSettingsByteArray =
        new byte[tfliteSettingslistEntry.tfliteSettings().getByteBuffer().remaining()];
    tfliteSettingslistEntry.tfliteSettings().getByteBuffer().get(tfliteSettingsByteArray);

    byte[] accuracyResultsByteArray =
        accuracyBenchmarkNativeRun(
            tfliteSettingsByteArray, modelFd, modelOffset, modelSize, resultPath);
    ByteBuffer byteBuffer = ByteBuffer.wrap(accuracyResultsByteArray);
    return BenchmarkEvent.getRootAsBenchmarkEvent(byteBuffer);
  }

  /**
   * Loads the input TFLiteSettings JSON files into TfLiteSettingsListEntry instances.
   *
   * <p>If the number of input TFLiteSettings JSON files is 1, we add one default entry at the
   * beginning as reference. The default entry contains a dummy TFLiteSettings structure, which lets
   * the interpreter to apply the default acceleration.
   */
  public static List<TfLiteSettingsListEntry> loadTfLiteSettingsList(String[] jsonFilePaths) {
    List<TfLiteSettingsListEntry> tfliteSettingsList = new ArrayList<>();
    if (jsonFilePaths.length == 1) {
      FlatBufferBuilder tfliteSettingsBuilder = new FlatBufferBuilder();
      TFLiteSettings.startTFLiteSettings(tfliteSettingsBuilder);
      int tfliteSettingsOffset = TFLiteSettings.endTFLiteSettings(tfliteSettingsBuilder);
      tfliteSettingsBuilder.finish(tfliteSettingsOffset);
      tfliteSettingsList.add(
          TfLiteSettingsListEntry.create(
              TFLiteSettings.getRootAsTFLiteSettings(tfliteSettingsBuilder.dataBuffer()),
              "default_delegate"));
    }
    for (String jsonFilePath : jsonFilePaths) {
      byte[] tfliteSettingsByteArray = loadTfLiteSettingsJsonNative(jsonFilePath);
      if (tfliteSettingsByteArray == null || tfliteSettingsByteArray.length == 0) {
        Log.e(TAG, "Failed to load TFLiteSetting from JSON file " + jsonFilePath);
        return new ArrayList<>();
      }

      ByteBuffer byteBuffer = ByteBuffer.wrap(tfliteSettingsByteArray);
      tfliteSettingsList.add(
          TfLiteSettingsListEntry.create(
              TFLiteSettings.getRootAsTFLiteSettings(byteBuffer), jsonFilePath));
    }
    return tfliteSettingsList;
  }

  private static native byte[] latencyBenchmarkNativeRun(
      String[] args, byte[] tfliteSettings, String tfliteSettingsPath);

  private static native byte[] accuracyBenchmarkNativeRun(
      byte[] tfliteSettings, int modelFd, long modelOffset, long modelSize, String resultPath);

  private static native byte[] loadTfLiteSettingsJsonNative(String jsonFilePath);
}
