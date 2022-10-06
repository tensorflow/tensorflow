/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.annotations.UsedByReflection;

/**
 * {@link Delegate} for GPU inference.
 *
 * <p>Note: When calling {@code Interpreter.Options.addDelegate()} and {@code Interpreter.run()},
 * the caller must have an {@code EGLContext} in the <b>current thread</b> and {@code
 * Interpreter.run()} must be called from the same {@code EGLContext}. If an {@code EGLContext} does
 * not exist, the delegate will internally create one, but then the developer must ensure that
 * {@code Interpreter.run()} is always called from the same thread in which {@code
 * Interpreter.Options.addDelegate()} was called.
 */
@UsedByReflection("TFLiteSupport/model/GpuDelegateProxy")
public class GpuDelegate implements Delegate {

  @Nullable
  private static final Throwable LOAD_LIBRARY_EXCEPTION;
  private static final long INVALID_DELEGATE_HANDLE = 0;
  private static final String TFLITE_GPU_LIB = "tensorflowlite_gpu_jni";
  private static volatile boolean isInit = false;

  private long delegateHandle;

  @UsedByReflection("GpuDelegateFactory")
  public GpuDelegate(GpuDelegateFactory.Options options) {
    init();
    delegateHandle =
        createDelegate(
            options.isPrecisionLossAllowed(),
            options.areQuantizedModelsAllowed(),
            options.getInferencePreference(),
            options.getSerializationDir(),
            options.getModelToken());
  }

  @UsedByReflection("TFLiteSupport/model/GpuDelegateProxy")
  public GpuDelegate() {
    this(new GpuDelegateFactory.Options());
  }

  /**
   * Inherits from {@link GpuDelegateFactory.Options} for compatibility with existing code.
   *
   * @deprecated Use {@link GpuDelegateFactory.Options} instead.
   */
  @Deprecated
  public static class Options extends GpuDelegateFactory.Options {}

  @Override
  public long getNativeHandle() {
    return delegateHandle;
  }

  /**
   * Frees TFLite resources in C runtime.
   *
   * <p>User is expected to call this method explicitly.
   */
  @Override
  public void close() {
    if (delegateHandle != INVALID_DELEGATE_HANDLE) {
      deleteDelegate(delegateHandle);
      delegateHandle = INVALID_DELEGATE_HANDLE;
    }
  }

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
  private static void init() {
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

  private static native void nativeDoNothing();

  private static native long createDelegate(
      boolean precisionLossAllowed,
      boolean quantizedModelsAllowed,
      int preference,
      String serializationDir,
      String modelToken);

  private static native void deleteDelegate(long delegateHandle);
}
