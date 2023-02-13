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

import java.util.Collections;
import java.util.Map;
import org.json.JSONException;
import org.json.JSONObject;
import tflite.Delegate;

/**
 * Helper class to store the raw performance results from the native layer.
 *
 * <p>An instance of {@link RawDelegateMetricsEntry} corresponds with each tested model and each
 * delegate.
 */
final class RawDelegateMetricsEntry {
  // The name of the delegate. The available names are listed in
  // tensorflow/tensorflow/lite/experimental/acceleration/configuration/configuration.proto
  // TODO(b/267431570): consider replacing the field with an Enum value.
  private final String delegateName;
  // Specifies the path to the delegate settings file.
  private final String path;
  // If {@link isTestTarget} is set to {@code true}, the metrics are gathered from the benchmark run
  // with the test target delegate.
  private final boolean isTestTarget;
  private final Map<String, Float> metrics;

  private RawDelegateMetricsEntry(
      int delegate, String path, boolean isTestTarget, Map<String, Float> metrics) {
    this.delegateName = Delegate.name(delegate);
    this.path = path;
    this.isTestTarget = isTestTarget;
    this.metrics = metrics;
  }

  /* Returns the name of the delegate. */
  String delegateName() {
    return delegateName;
  }

  /* Returns the path to the delegate settings file. */
  String path() {
    return path;
  }

  /* Returns whether the delegate is the test target. */
  boolean isTestTarget() {
    return isTestTarget;
  }

  Map<String, Float> metrics() {
    return Collections.unmodifiableMap(metrics);
  }

  /**
   * Returns an identifier for the delegate. The idenfier consists of the delegate type and the path
   * to the delegate settings file.
   */
  String delegateIdentifier() {
    return delegateName + " (" + path + ")";
  }

  JSONObject toJsonObject() throws JSONException {
    JSONObject jsonObject = new JSONObject();
    jsonObject.put("delegate_name", delegateName);
    jsonObject.put("path", path);
    jsonObject.put("is_test_target", isTestTarget);
    JSONObject metricsObject = new JSONObject();
    for (Map.Entry<String, Float> entry : metrics.entrySet()) {
      metricsObject.put(entry.getKey(), entry.getValue());
    }
    jsonObject.put("metrics", metricsObject);
    return jsonObject;
  }

  static RawDelegateMetricsEntry create(
      int delegate, String path, boolean isTestTarget, Map<String, Float> metrics) {
    return new RawDelegateMetricsEntry(delegate, path, isTestTarget, metrics);
  }
}
