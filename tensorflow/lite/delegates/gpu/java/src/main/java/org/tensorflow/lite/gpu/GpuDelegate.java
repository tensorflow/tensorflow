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
import org.tensorflow.lite.Tensor;

/** {@link Delegate} for GPU inference. */
public class GpuDelegate implements Delegate, Closeable {

  private static final long INVALID_DELEGATE_HANDLE = 0;
  private static final String TFLITE_GPU_LIB = "tensorflowlite_gpu_jni";

  private long delegateHandle;

  /** Shader compilation options. */
  public static final class CompileOptions {
    public CompileOptions() {}

    /** Delegate chooses fastest GL object type to represent tensors (default). */
    public static final int GL_OBJECT_TYPE_FASTEST = 0;
    /**
     * Delegate uses GL textures to represent tensors, which works faster on Adreno-based devices,
     * but may use more memory.
     */
    public static final int GL_OBJECT_TYPE_TEXTURE = 1;
    /** Delegate uses GL shader storage buffer objects to represent tensors. */
    public static final int GL_OBJECT_TYPE_BUFFER = 2;

    /**
     * Sets whether precision loss is allowed.
     *
     * @param precisionLossAllowed When `true` (default), the GPU may quantify tensors, downcast
     *     values, process in FP16. When `false`, computations are carried out in 32-bit floating
     *     point.
     */
    public CompileOptions setPrecisionLossAllowed(boolean precisionLossAllowed) {
      this.precisionLossAllowed = precisionLossAllowed;
      return this;
    }

    /**
     * Sets whether dynamic batch is enabled.
     *
     * @param dynamicBatchEnabled When `false` (default), dynamic batching is disabled and
     *     input/output tensors must have a batch size of 1 (probably what you want, unless you use
     *     LSTMs). When `true`, enables dynamic batching and input/output tensor can have a batch
     *     size greater than 1.
     */
    public CompileOptions setDynamicBatchEnabled(boolean dynamicBatchEnabled) {
      this.dynamicBatchEnabled = dynamicBatchEnabled;
      return this;
    }

    /**
     * Sets the preferred GL object type for tensor representation
     *
     * @param preferredGlObjectType One of `GL_OBJECT_TYPE_FASTEST` (default),
     *     `GL_OBJECT_TYPE_TEXTURE`, `GL_OBJECT_TYPE_BUFFER`.
     */
    public CompileOptions setPreferredGlObjectType(int preferredGlObjectType) {
      this.preferredGlObjectType = preferredGlObjectType;
      return this;
    }

    boolean precisionLossAllowed = true;
    boolean dynamicBatchEnabled = false;
    int preferredGlObjectType = GL_OBJECT_TYPE_FASTEST;
  }

  /** Delegate options. */
  public static final class Options {
    public Options() {}

    private static final CompileOptions DEFAULT_COMPILE_OPTIONS = new CompileOptions();

    /**
     * Sets the shader compilation options to be used by the delegate.
     *
     * @param compileOptions the {@link CompileOptions} to use.
     */
    public Options setCompileOptions(CompileOptions compileOptions) {
      this.compileOptions = compileOptions != null ? compileOptions : DEFAULT_COMPILE_OPTIONS;
      return this;
    }

    CompileOptions compileOptions = DEFAULT_COMPILE_OPTIONS;
  }

  public GpuDelegate(Options options) {
    delegateHandle =
        createDelegate(
            options.compileOptions.precisionLossAllowed,
            options.compileOptions.dynamicBatchEnabled,
            options.compileOptions.preferredGlObjectType);
  }

  public GpuDelegate() {
    this(new Options());
  }

  /**
   * Advanced: Binds a GL SSBO to an input or an output tensor in the initialized delegate.
   *
   * <p>The bound buffer should have sufficient storage to accommodate all elements of the tensor.
   *
   * <p><b>Note:</b> This method must be called *before* calling the delegate instance is installed
   * in the {@link Interpreter}.
   *
   * <p>WARNING: This is an experimental API and subject to change.
   *
   * @param tensor The input or output {@link Tensor} to bind to the buffer object.
   * @param ssbo The GL buffer object to bind to the tensor. See also {@link
   *     Interpreter.Options#setAllowBufferHandleOutput()} for details on allowing zero-copy output
   *     when GL textures are bound to output tensors.
   * @return Whether the operation succeeded.
   */
  public boolean bindGlBufferToTensor(Tensor tensor, int ssbo) {
    return bindGlBufferToTensor(delegateHandle, tensor.index(), ssbo);
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
      boolean precisionLossAllowed, boolean dynamicBatchEnabled, int preferredGlObjectType);

  private static native void deleteDelegate(long delegateHandle);

  private static native boolean bindGlBufferToTensor(
      long delegateHandle, int tensorIndex, int ssbo);
}
