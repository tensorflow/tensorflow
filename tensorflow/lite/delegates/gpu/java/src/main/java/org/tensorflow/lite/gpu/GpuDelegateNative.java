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

package org.tensorflow.lite.gpu;

import androidx.annotation.Nullable;

/**
 * Static helper class to load the native library while gracefully handling the case where native
 * method implementations are linked into a different library.
 */
class GpuDelegateNative {

  @Nullable static final Throwable LOAD_LIBRARY_EXCEPTION;
  private static final String TFLITE_GPU_LIB = "tensorflowlite_gpu_jni";
  private static volatile boolean isInit = false;

  static {
    Throwable exception = null;
    try {
      System.loadLibrary(TFLITE_GPU_LIB);
    } catch (UnsatisfiedLinkError e) {
      exception = e;
    }
    LOAD_LIBRARY_EXCEPTION = exception;
  }

  /**
   * Ensure the GpuDelegate native library has been loaded.
   *
   * <p>If unsuccessful, throws an UnsatisfiedLinkError with the appropriate error message.
   */
  static void init() {
    if (isInit) {
      return;
    }

    try {
      // Try to invoke a native method (which itself does nothing) to ensure that native libs are
      // available.
      // This code is thread safe without synchronization, as multiple concurrent callers will
      // either throw an exception without setting this value or set it to true several times.
      nativeDoNothing();
      isInit = true;
    } catch (UnsatisfiedLinkError originalUnsatisfiedLinkError) {
      // Prefer logging the original library loading exception if native methods are unavailable.
      Throwable exceptionToLog =
          LOAD_LIBRARY_EXCEPTION != null ? LOAD_LIBRARY_EXCEPTION : originalUnsatisfiedLinkError;
      UnsatisfiedLinkError exceptionToThrow =
          new UnsatisfiedLinkError(
              "Failed to load native GpuDelegate methods. Check that the correct native"
                  + " libraries are present, and, if using a custom native library, have been"
                  + " properly loaded via System.loadLibrary():\n"
                  + "  "
                  + exceptionToLog);
      exceptionToThrow.initCause(originalUnsatisfiedLinkError);
      exceptionToThrow.addSuppressed(exceptionToLog);
      throw exceptionToThrow;
    }
  }

  private GpuDelegateNative() {}

  private static native void nativeDoNothing();
}
