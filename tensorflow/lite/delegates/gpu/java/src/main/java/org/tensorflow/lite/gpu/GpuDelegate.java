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

  private static final long INVALID_DELEGATE_HANDLE = 0;
  private static final String TFLITE_GPU_LIB = "tensorflowlite_gpu_jni";

  private long delegateHandle;

  @UsedByReflection("GpuDelegateFactory")
  public GpuDelegate(GpuDelegateFactory.Options options) {
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
    System.loadLibrary(TFLITE_GPU_LIB);
  }

  private static native long createDelegate(
      boolean precisionLossAllowed,
      boolean quantizedModelsAllowed,
      int preference,
      String serializationDir,
      String modelToken);

  private static native void deleteDelegate(long delegateHandle);
}
