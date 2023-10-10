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

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;

/** Static utility methods for loading the TensorFlowLite runtime and native code. */
public final class TensorFlowLite {
  // We use Java logging here (java.util.logging), rather than Android logging (android.util.Log),
  // to avoid unnecessary platform dependencies. This also makes unit testing simpler and faster,
  // since we can use plain Java tests rather than needing to use Robolectric (android_local_test).
  //
  // WARNING: some care is required when using Java logging on Android.  In particular, avoid
  // logging with severity levels lower than "INFO", since the default Java log handler on Android
  // will discard those, and avoid logging messages with parameters (call String.format instead),
  // since the default Java log handler on Android only logs the raw message string and doesn't
  // apply the parameters.
  private static final Logger logger = Logger.getLogger(InterpreterApi.class.getName());

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
          logger.info("Loaded native library: " + libName);
          break;
        } catch (UnsatisfiedLinkError e) {
          logger.info("Didn't load native library: " + libName);
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

  /** Returns the version of the specified TensorFlowLite runtime. */
  public static String runtimeVersion(TfLiteRuntime runtime) {
    return getFactory(runtime, "org.tensorflow.lite.TensorFlowLite", "runtimeVersion")
        .runtimeVersion();
  }

  /** Returns the version of the default TensorFlowLite runtime. */
  public static String runtimeVersion() {
    return runtimeVersion(null);
  }

  /**
   * Returns the version of the TensorFlowLite model schema that is supported by the specified
   * TensorFlowLite runtime.
   */
  public static String schemaVersion(TfLiteRuntime runtime) {
    return getFactory(runtime, "org.tensorflow.lite.TensorFlowLite", "schemaVersion")
        .schemaVersion();
  }

  /**
   * Returns the version of the TensorFlowLite model schema that is supported by the default
   * TensorFlowLite runtime.
   */
  public static String schemaVersion() {
    return schemaVersion(null);
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

  /** Encapsulates the use of reflection to find an available TF Lite runtime. */
  private static class PossiblyAvailableRuntime {
    private final InterpreterFactoryApi factory;
    private final Exception exception;

    /**
     * @param namespace: "org.tensorflow.lite" or "com.google.android.gms.tflite".
     * @param category: "application" or "system".
     */
    PossiblyAvailableRuntime(String namespace, String category) {
      InterpreterFactoryApi factory = null;
      Exception exception = null;
      try {
        Class<?> clazz = Class.forName(namespace + ".InterpreterFactoryImpl");
        Constructor<?> factoryConstructor = clazz.getDeclaredConstructor();
        factoryConstructor.setAccessible(true);
        factory = (InterpreterFactoryApi) factoryConstructor.newInstance();
        if (factory != null) {
          logger.info(String.format("Found %s TF Lite runtime client in %s", category, namespace));
        } else {
          logger.warning(
              String.format("Failed to construct TF Lite runtime client from %s", namespace));
        }
      } catch (ClassNotFoundException
          | IllegalAccessException
          | IllegalArgumentException
          | InstantiationException
          | InvocationTargetException
          | NoSuchMethodException
          | SecurityException e) {
        logger.info(
            String.format("Didn't find %s TF Lite runtime client in %s", category, namespace));
        exception = e;
      }
      this.exception = exception;
      this.factory = factory;
    }
    /**
     * @return the InterpreterFactoryApi for this runtime, or null if this runtime wasn't found.
     */
    public InterpreterFactoryApi getFactory() {
      return factory;
    }
    /**
     * @return The exception that occurred when trying to find this runtime, if any, or null.
     */
    public Exception getException() {
      return exception;
    }
  }

  // We use static members here for caching, to ensure that we only do the reflective lookups once
  // and then afterwards re-use the previously computed results.
  //
  // We put these static members in nested static classes to ensure that Java will
  // delay the initialization of these static members until their respective first use;
  // that's needed to ensure that we only log messages about TF Lite runtime not found
  // for TF Lite runtimes that the application actually tries to use.
  private static class RuntimeFromSystem {
    static final PossiblyAvailableRuntime TFLITE =
        new PossiblyAvailableRuntime("com.google.android.gms.tflite", "system");
  }

  private static class RuntimeFromApplication {
    static final PossiblyAvailableRuntime TFLITE =
        new PossiblyAvailableRuntime("org.tensorflow.lite", "application");
  }

  // We log at most once for each different options.runtime value.
  private static final AtomicBoolean[] haveLogged =
      new AtomicBoolean[TfLiteRuntime.values().length];

  static {
    for (int i = 0; i < TfLiteRuntime.values().length; i++) {
      haveLogged[i] = new AtomicBoolean();
    }
  }

  static InterpreterFactoryApi getFactory(TfLiteRuntime runtime) {
    return getFactory(runtime, "org.tensorflow.lite.InterpreterApi.Options", "setRuntime");
  }

  /**
   * Internal method for finding the TF Lite runtime implementation.
   *
   * @param className Class name for method to mention in exception messages.
   * @param methodName Method name for method to mention in exception messages.
   */
  private static InterpreterFactoryApi getFactory(
      TfLiteRuntime runtime, String className, String methodName) {
    Exception exception = null;
    if (runtime == null) {
      runtime = TfLiteRuntime.FROM_APPLICATION_ONLY;
    }
    if (runtime == TfLiteRuntime.PREFER_SYSTEM_OVER_APPLICATION
        || runtime == TfLiteRuntime.FROM_SYSTEM_ONLY) {
      if (RuntimeFromSystem.TFLITE.getFactory() != null) {
        if (!haveLogged[runtime.ordinal()].getAndSet(true)) {
          logger.info(
              String.format(
                  "TfLiteRuntime.%s: "
                      + "Using system TF Lite runtime client from com.google.android.gms",
                  runtime.name()));
        }
        return RuntimeFromSystem.TFLITE.getFactory();
      } else {
        exception = RuntimeFromSystem.TFLITE.getException();
      }
    }
    if (runtime == TfLiteRuntime.PREFER_SYSTEM_OVER_APPLICATION
        || runtime == TfLiteRuntime.FROM_APPLICATION_ONLY) {
      if (RuntimeFromApplication.TFLITE.getFactory() != null) {
        if (!haveLogged[runtime.ordinal()].getAndSet(true)) {
          logger.info(
              String.format(
                  "TfLiteRuntime.%s: "
                      + "Using application TF Lite runtime client from org.tensorflow.lite",
                  runtime.name()));
        }
        return RuntimeFromApplication.TFLITE.getFactory();
      } else {
        if (exception == null) {
          exception = RuntimeFromApplication.TFLITE.getException();
        } else if (exception.getSuppressed().length == 0) {
          exception.addSuppressed(RuntimeFromApplication.TFLITE.getException());
        }
      }
    }
    String message;
    switch (runtime) {
      case FROM_APPLICATION_ONLY:
        message =
            String.format(
                "You should declare a build dependency on org.tensorflow.lite:tensorflow-lite,"
                    + " or call .%s with a value other than TfLiteRuntime.FROM_APPLICATION_ONLY"
                    + " (see docs for %s#%s(TfLiteRuntime)).",
                methodName, className, methodName);
        break;
      case FROM_SYSTEM_ONLY:
        message =
            String.format(
                "You should declare a build dependency on"
                    + " com.google.android.gms:play-services-tflite-java,"
                    + " or call .%s with a value other than TfLiteRuntime.FROM_SYSTEM_ONLY "
                    + " (see docs for %s#%s).",
                methodName, className, methodName);
        break;
      default:
        message =
            "You should declare a build dependency on"
                + " org.tensorflow.lite:tensorflow-lite or"
                + " com.google.android.gms:play-services-tflite-java";
        break;
    }
    throw new IllegalStateException(
        "Couldn't find TensorFlow Lite runtime's InterpreterFactoryImpl class --"
            + " make sure your app links in the right TensorFlow Lite runtime. "
            + message,
        exception);
  }
}
