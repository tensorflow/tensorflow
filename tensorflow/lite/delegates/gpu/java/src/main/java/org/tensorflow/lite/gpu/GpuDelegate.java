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

import java.io.Closeable;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.annotations.UsedByReflection;

/**
 * {@link Delegate} for GPU inference.
 *
 * <p>Note: When calling {@code Interpreter.modifyGraphWithDelegate()}/ {@code
 * Interpreter.Options.addDelegate()} and {@code Interpreter.run()}, the caller must have an {@code
 * EGLContext} in the <b>current thread</b> and {@code Interpreter.run()} must be called from the
 * same {@code EGLContext}. If an {@code EGLContext} does not exist, the delegate will internally
 * create one, but then the developer must ensure that {@code Interpreter.run()} is always called
 * from the same thread in which {@code Interpreter.modifyGraphWithDelegate()} was called.
 */
@UsedByReflection("TFLiteSupport/model/GpuDelegateProxy")
public class GpuDelegate implements Delegate, Closeable {

  private static final long INVALID_DELEGATE_HANDLE = 0;
  private static final String TFLITE_GPU_LIB = "tensorflowlite_gpu_jni";

  private long delegateHandle;

  /** Delegate options. */
  public static final class Options {
    public Options() {}

    /**
     * Delegate will be used only once, therefore, bootstrap/init time should be taken into account.
     */
    public static final int INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER = 0;

    /**
     * Prefer maximizing the throughput. Same delegate will be used repeatedly on multiple inputs.
     */
    public static final int INFERENCE_PREFERENCE_SUSTAINED_SPEED = 1;

    /**
     * Sets whether precision loss is allowed.
     *
     * @param precisionLossAllowed When `true` (default), the GPU may quantify tensors, downcast
     *     values, process in FP16. When `false`, computations are carried out in 32-bit floating
     *     point.
     */
    public Options setPrecisionLossAllowed(boolean precisionLossAllowed) {
      this.precisionLossAllowed = precisionLossAllowed;
      return this;
    }

    /**
     * Enables running quantized models with the delegate. Defaults to false.
     *
     * <p>WARNING: This is an experimental API and subject to change.
     *
     * @param quantizedModelsAllowed When {@code true} (default), the GPU may run quantized models.
     */
    public Options setQuantizedModelsAllowed(boolean quantizedModelsAllowed) {
      this.quantizedModelsAllowed = quantizedModelsAllowed;
      return this;
    }

    /**
     * Sets the inference preference for precision/compilation/runtime tradeoffs.
     *
     * @param preference One of `INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER` (default),
     *     `INFERENCE_PREFERENCE_SUSTAINED_SPEED`.
     */
    public Options setInferencePreference(int preference) {
      this.inferencePreference = preference;
      return this;
    }

    boolean precisionLossAllowed = true;
    boolean quantizedModelsAllowed = true;
    int inferencePreference = INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
  }

  public GpuDelegate(Options options) {
    delegateHandle =
        createDelegate(
            options.precisionLossAllowed,
            options.quantizedModelsAllowed,
            options.inferencePreference);
  }

  @UsedByReflection("TFLiteSupport/model/GpuDelegateProxy")
  public GpuDelegate() {
    this(new Options());
  }

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
      boolean precisionLossAllowed, boolean quantizedModelsAllowed, int preference);

  private static native void deleteDelegate(long delegateHandle);
}
