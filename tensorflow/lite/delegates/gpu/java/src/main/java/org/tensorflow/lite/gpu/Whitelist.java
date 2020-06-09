/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

/**
 * GPU Delegate Whitelisting data.
 *
 * <p>The GPU delegate is not supported on all Android devices, due to differences in available
 * OpenGL versions, driver features, and device resources. This class provides information on
 * whether the GPU delegate is suitable for the current device.
 *
 * <p>This API is experimental and subject to change.
 *
 * <p><b>WARNING:</b> the whitelist is constructed from testing done on a limited set of models. You
 * should plan to verify that your own model(s) work.
 *
 * <p>Example usage:
 *
 * <pre>{@code
 * Interpreter.Options options = new Interpreter.Options();
 * try (Whitelist whitelist = new Whitelist()) {
 *   if (whitelist.isDelegateSupportedOnThisDevice()) {
 *     GpuDelegate.Options delegateOptions = whitelist.getBestOptionsForThisDevice();
 *     gpuDelegate = new GpuDelegate(delegateOptions):
 *     options.addDelegate(gpuDelegate);
 *   }
 * }
 * Interpreter interpreter = new Interpreter(modelBuffer, options);
 * }</pre>
 */
public class Whitelist implements Closeable {

  private static final long INVALID_WHITELIST_HANDLE = 0;
  private static final String TFLITE_GPU_LIB = "tensorflowlite_gpu_jni";

  private long whitelistHandle = INVALID_WHITELIST_HANDLE;

  /** Whether the GPU delegate is supported on this device. */
  public boolean isDelegateSupportedOnThisDevice() {
    if (whitelistHandle == INVALID_WHITELIST_HANDLE) {
      throw new IllegalStateException("Trying to query a closed whitelist.");
    }
    return nativeIsDelegateSupportedOnThisDevice(whitelistHandle);
  }

  /** What options should be used for the GPU delegate. */
  public GpuDelegate.Options getBestOptionsForThisDevice() {
    // For forward compatibility, when the whitelist contains more information.
    return new GpuDelegate.Options();
  }

  public Whitelist() {
    whitelistHandle = createWhitelist();
  }

  /**
   * Frees TFLite resources in C runtime.
   *
   * <p>User is expected to call this method explicitly.
   */
  @Override
  public void close() {
    if (whitelistHandle != INVALID_WHITELIST_HANDLE) {
      deleteWhitelist(whitelistHandle);
      whitelistHandle = INVALID_WHITELIST_HANDLE;
    }
  }

  static {
    System.loadLibrary(TFLITE_GPU_LIB);
  }

  private static native long createWhitelist();

  private static native void deleteWhitelist(long handle);

  private static native boolean nativeIsDelegateSupportedOnThisDevice(long handle);
}
