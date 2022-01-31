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

  private static final String[][] TFLITE_RUNTIME_LIBNAMES =
      new String[][] {
        // We load the first library that we find in each group.
        new String[] {
          // Regular TF Lite.
          "tensorflowlite_jni", // Full library, including experimental features.
          "tensorflowlite_jni_stable", // Subset excluding experimental features.
        },
        new String[] {
          // TF Lite from system.
          "tensorflowlite_jni_gms_client"
        }
      };

  private static final Throwable LOAD_LIBRARY_EXCEPTION;
  private static volatile boolean isInit = false;

  static {
    // Attempt to load the TF Lite runtime's JNI library, trying each alternative name in turn.
    // If unavailable, catch and save the exception(s); the client may choose to link the native
    // deps into their own custom native library, so it's not an error if the default library names
    // can't be loaded.
    Throwable loadLibraryException = null;
    for (String[] group : TFLITE_RUNTIME_LIBNAMES) {
      for (String libName : group) {
        try {
          System.loadLibrary(libName);
          break;
        } catch (UnsatisfiedLinkError e) {
          if (loadLibraryException == null) {
            loadLibraryException = e;
          } else {
            loadLibraryException.addSuppressed(e);
          }
        }
      }
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
    return InterpreterApi.getFactory(new InterpreterApi.Options()).runtimeVersion();
  }

  /** Returns the version of the underlying TensorFlowLite model schema. */
  public static String schemaVersion() {
    return InterpreterApi.getFactory(new InterpreterApi.Options()).schemaVersion();
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
      // Try to invoke a native method (which itself does nothing) to ensure that native libs are
      // available.
      nativeDoNothing();
      isInit = true;
    } catch (UnsatisfiedLinkError e) {
      // Prefer logging the original library loading exception if native methods are unavailable.
      Throwable exceptionToLog = LOAD_LIBRARY_EXCEPTION != null ? LOAD_LIBRARY_EXCEPTION : e;
      UnsatisfiedLinkError exceptionToThrow =
          new UnsatisfiedLinkError(
              "Failed to load native TensorFlow Lite methods. Check that the correct native"
                  + " libraries are present, and, if using a custom native library, have been"
                  + " properly loaded via System.loadLibrary():\n"
                  + "  "
                  + exceptionToLog);
      exceptionToThrow.initCause(e);
      throw exceptionToThrow;
    }
  }

  private static native void nativeDoNothing();
}
