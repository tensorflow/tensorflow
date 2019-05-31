/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.SavedModelBundle}. */
@RunWith(JUnit4.class)
public class SavedModelBundleTest {

  private static final String SAVED_MODEL_PATH =
      "tensorflow/cc/saved_model/testdata/half_plus_two/00000123";

  @Test
  public void load() {
    try (SavedModelBundle bundle = SavedModelBundle.load(SAVED_MODEL_PATH, "serve")) {
      assertNotNull(bundle.session());
      assertNotNull(bundle.graph());
      assertNotNull(bundle.metaGraphDef());
    }
  }

  @Test
  public void loadNonExistentBundle() {
    try {
      SavedModelBundle bundle = SavedModelBundle.load("__BAD__", "serve");
      bundle.close();
      fail("not expected");
    } catch (org.tensorflow.TensorFlowException e) {
      // expected exception
      assertTrue(e.getMessage().contains("Could not find SavedModel"));
    }
  }

  @Test
  public void loader() {
    try (SavedModelBundle bundle = SavedModelBundle.loader(SAVED_MODEL_PATH)
        .withTags("serve")
        .withConfigProto(sillyConfigProto())
        .withRunOptions(sillyRunOptions())
        .load()) {
      assertNotNull(bundle.session());
      assertNotNull(bundle.graph());
      assertNotNull(bundle.metaGraphDef());
    }
  }

  private static byte[] sillyRunOptions() {
    // Ideally this would use the generated Java sources for protocol buffers
    // and end up with something like the snippet below. However, generating
    // the Java files for the .proto files in tensorflow/core:protos_all is
    // a bit cumbersome in bazel until the proto_library rule is setup.
    //
    // See https://github.com/bazelbuild/bazel/issues/52#issuecomment-194341866
    // https://github.com/bazelbuild/rules_go/pull/121#issuecomment-251515362
    // https://github.com/bazelbuild/rules_go/pull/121#issuecomment-251692558
    //
    // For this test, for now, the use of specific bytes suffices.
    return new byte[] {0x08, 0x03};
    /*
    return org.tensorflow.framework.RunOptions.newBuilder()
        .setTraceLevel(RunOptions.TraceLevel.FULL_TRACE)
        .build()
        .toByteArray();
    */
  }

  public static byte[] sillyConfigProto() {
    // Ideally this would use the generated Java sources for protocol buffers
    // and end up with something like the snippet below. However, generating
    // the Java files for the .proto files in tensorflow/core:protos_all is
    // a bit cumbersome in bazel until the proto_library rule is setup.
    //
    // See https://github.com/bazelbuild/bazel/issues/52#issuecomment-194341866
    // https://github.com/bazelbuild/rules_go/pull/121#issuecomment-251515362
    // https://github.com/bazelbuild/rules_go/pull/121#issuecomment-251692558
    //
    // For this test, for now, the use of specific bytes suffices.
    return new byte[] {0x10, 0x01, 0x28, 0x01};
    /*
    return org.tensorflow.framework.ConfigProto.newBuilder()
        .setInterOpParallelismThreads(1)
        .setIntraOpParallelismThreads(1)
        .build()
        .toByteArray();
     */
  }
}
