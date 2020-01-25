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

  private static final Throwable LOAD_LIBRARY_EXCEPTION;
  private static volatile boolean isInit = false;

  static {
    // Attempt to load the default native libraries. If unavailable, cache the exception; the client
    // may choose to link the native deps into their own custom native library.
    Throwable loadLibraryException = null;
    try {
      System.loadLibrary(LIBNAME);
    } catch (UnsatisfiedLinkError e) {
      loadLibraryException = e;
    }
    LOAD_LIBRARY_EXCEPTION = loadLibraryException;
  }

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
  public static String runtimeVersion() {
    init();
    return nativeRuntimeVersion();
  }

  /** Returns the version of the underlying TensorFlowLite model schema. */
  public static String schemaVersion() {
    init();
    return nativeSchemaVersion();
  }

  /**
   * Ensure the TensorFlowLite native library has been loaded.
   *
   * <p>If unsuccessful, throws an UnsatisfiedLinkError with the appropriate error message.
   */
  public static void init() {
    if (isInit) {
      return;
    }

    try {
      // Try to invoke a native method (the method itself doesn't really matter) to ensure that
      // native libs are available.
      nativeRuntimeVersion();
      isInit = true;
    } catch (UnsatisfiedLinkError e) {
      // Prefer logging the original library loading exception if native methods are unavailable.
      Throwable exceptionToLog = LOAD_LIBRARY_EXCEPTION != null ? LOAD_LIBRARY_EXCEPTION : e;
      throw new UnsatisfiedLinkError(
          "Failed to load native TensorFlow Lite methods. Check "
              + "that the correct native libraries are present, and, if using "
              + "a custom native library, have been properly loaded via System.loadLibrary():\n  "
              + exceptionToLog);
    }
  }

  public static native String nativeRuntimeVersion();

  public static native String nativeSchemaVersion();
}
