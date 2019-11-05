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

/** Static utility methods loading the TensorFlowLite runtime. */
public final class TensorFlowLite {

  private static final String LIBNAME = "tensorflowlite_jni";

  private TensorFlowLite() {}

  /**
   * Returns the version of the underlying TensorFlowLite model schema.
   *
   * @deprecated Prefer using {@link #runtimeVersion() or #schemaVersion()}.
   */
  @Deprecated
  public static String version() {
    return schemaVersion();
  }

  /** Returns the version of the underlying TensorFlowLite runtime. */
  public static native String runtimeVersion();

  /** Returns the version of the underlying TensorFlowLite model schema. */
  public static native String schemaVersion();

  /**
   * Load the TensorFlowLite runtime C library.
   *
   * @hide
   */
  public static boolean init() {
    try {
      System.loadLibrary(LIBNAME);
      return true;
    } catch (UnsatisfiedLinkError e) {
      System.err.println("TensorFlowLite: failed to load native library: " + e);
      return false;
    }
  }

  static {
    init();
  }
}
