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

package org.tensorflow.lite.experimental;

import java.io.Closeable;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Tensor;

/** {@link Delegate} for GPU inference. */
public class GpuDelegate implements Delegate, Closeable {

  private static final long INVALID_DELEGATE_HANDLE = 0;
  private static final String TFLITE_GPU_LIB = "tensorflowlite_gpu_jni";

  private long delegateHandle;

  public GpuDelegate() {
    delegateHandle = createDelegate();
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

  private static native long createDelegate();

  private static native void deleteDelegate(long delegateHandle);

  private static native boolean bindGlBufferToTensor(
      long delegateHandle, int tensorIndex, int ssbo);
}
