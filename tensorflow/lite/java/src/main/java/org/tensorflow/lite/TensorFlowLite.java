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

  private static final String PRIMARY_LIBNAME = "tensorflowlite_jni";
  private static final String FALLBACK_LIBNAME = "tensorflowlite_flex_jni";

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
   * Initialize tensorflow's libraries. This will throw an exception if used when TensorFlow isn't
   * linked in.
   */
  static native void initTensorFlow();

  /**
   * Load the TensorFlowLite runtime C library.
   */
  static boolean init() {
    Throwable primaryLibException;
    try {
      System.loadLibrary(PRIMARY_LIBNAME);
      return true;
    } catch (UnsatisfiedLinkError e) {
      primaryLibException = e;
    }

    try {
      System.loadLibrary(FALLBACK_LIBNAME);
      return true;
    } catch (UnsatisfiedLinkError e) {
      // If the fallback fails, log the error for the primary load instead.
      System.err.println(
          "TensorFlowLite: failed to load native library: " + primaryLibException.getMessage());
    }

    return false;
  }

  static {
    init();
  }
}
