/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;

// LINT.IfChange

/** Unit tests for {@link org.tensorflow.lite.TensorFlowLite}. */
@RunWith(JUnit4.class)
public final class TensorFlowLiteTest {

  @Before
  public void setUp() {
    TestInit.init();
  }

  @Test
  @SuppressWarnings("deprecation")
  public void testVersion() {
    assertThat(TensorFlowLite.version()).isEqualTo("3");
  }

  @Test
  public void testSchemaVersionForDefaultRuntime() {
    try {
      assertThat(TensorFlowLite.schemaVersion()).isEqualTo("3");
    } catch (IllegalStateException e) {
      assertThat(e).hasMessageThat().contains("org.tensorflow");
      assertThat(e).hasMessageThat().contains("tensorflow-lite");
      assertThat(e).hasMessageThat().doesNotContain("com.google.android.gms");
      assertThat(e).hasMessageThat().doesNotContain("play-services-tflite-java");
    }
  }

  @Test
  public void testSchemaVersionForSpecifiedRuntime() {
    assertThat(TensorFlowLite.schemaVersion(TfLiteRuntime.PREFER_SYSTEM_OVER_APPLICATION))
        .isEqualTo("3");
    try {
      assertThat(TensorFlowLite.schemaVersion(TfLiteRuntime.FROM_APPLICATION_ONLY)).isEqualTo("3");
    } catch (IllegalStateException e) {
      assertThat(e).hasMessageThat().contains("org.tensorflow");
      assertThat(e).hasMessageThat().contains("tensorflow-lite");
      assertThat(e).hasMessageThat().doesNotContain("com.google.android.gms");
      assertThat(e).hasMessageThat().doesNotContain("play-services-tflite-java");
      assertThat(e).hasMessageThat().contains("org.tensorflow.lite.TensorFlowLite#schemaVersion");
    }
    try {
      assertThat(TensorFlowLite.schemaVersion(TfLiteRuntime.FROM_SYSTEM_ONLY)).isEqualTo("3");
    } catch (IllegalStateException e) {
      assertThat(e).hasMessageThat().contains("com.google.android.gms");
      assertThat(e).hasMessageThat().contains("play-services-tflite-java");
      assertThat(e).hasMessageThat().doesNotContain("org.tensorflow:tensorflow-lite");
      assertThat(e).hasMessageThat().contains("org.tensorflow.lite.TensorFlowLite#schemaVersion");
    }
  }

  @Test
  public void testRuntimeVersionForDefaultRuntime() {
    // Unlike the schema version, which should almost never change, the runtime version can change
    // with some frequency, so simply ensure that it's non-empty and doesn't fail.
    try {
      assertThat(TensorFlowLite.runtimeVersion()).isNotEmpty();
    } catch (IllegalStateException e) {
      assertThat(e).hasMessageThat().contains("org.tensorflow");
      assertThat(e).hasMessageThat().contains("tensorflow-lite");
      assertThat(e).hasMessageThat().doesNotContain("com.google.android.gms");
      assertThat(e).hasMessageThat().doesNotContain("play-services-tflite-java");
      assertThat(e).hasMessageThat().contains("org.tensorflow.lite.TensorFlowLite#runtimeVersion");
    }
  }

  @Test
  public void testRuntimeVersionForSpecifiedRuntime() {
    // Unlike the schema version, which should almost never change, the runtime version can change
    // with some frequency, so simply ensure that it's non-empty and doesn't fail,
    // or that if it fails due to not having the appropriate runtime linked in,
    // it reports an appropriate error message.
    assertThat(TensorFlowLite.runtimeVersion(TfLiteRuntime.PREFER_SYSTEM_OVER_APPLICATION))
        .isNotEmpty();
    try {
      assertThat(TensorFlowLite.runtimeVersion(TfLiteRuntime.FROM_APPLICATION_ONLY)).isNotEmpty();
    } catch (IllegalStateException e) {
      assertThat(e).hasMessageThat().contains("org.tensorflow");
      assertThat(e).hasMessageThat().contains("tensorflow-lite");
      assertThat(e).hasMessageThat().doesNotContain("com.google.android.gms");
      assertThat(e).hasMessageThat().doesNotContain("play-services-tflite-java");
      assertThat(e).hasMessageThat().contains("org.tensorflow.lite.TensorFlowLite#runtimeVersion");
    }
    try {
      assertThat(TensorFlowLite.runtimeVersion(TfLiteRuntime.FROM_SYSTEM_ONLY)).isNotEmpty();
    } catch (IllegalStateException e) {
      assertThat(e).hasMessageThat().contains("com.google.android.gms");
      assertThat(e).hasMessageThat().contains("play-services-tflite-java");
      assertThat(e).hasMessageThat().doesNotContain("org.tensorflow:tensorflow-lite");
      assertThat(e).hasMessageThat().contains("org.tensorflow.lite.TensorFlowLite#runtimeVersion");
    }
  }
}

// LINT.ThenChange(../../../../../../BUILD:TensorFlowLiteTestShardCount)
