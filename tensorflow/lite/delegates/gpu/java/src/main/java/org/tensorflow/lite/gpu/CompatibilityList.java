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
 * GPU Delegate CompatibilityListing data.
 *
 * <p>The GPU delegate is not supported on all Android devices, due to differences in available
 * OpenGL versions, driver features, and device resources. This class provides information on
 * whether the GPU delegate is suitable for the current device.
 *
 * <p>This API is experimental and subject to change.
 *
 * <p><b>WARNING:</b> the compatibilityList is constructed from testing done on a limited set of
 * models. You should plan to verify that your own model(s) work.
 *
 * <p>Example usage:
 *
 * <pre>{@code
 * Interpreter.Options options = new Interpreter.Options();
 * try (CompatibilityList compatibilityList = new CompatibilityList()) {
 *   if (compatibilityList.isDelegateSupportedOnThisDevice()) {
 *     GpuDelegate.Options delegateOptions = compatibilityList.getBestOptionsForThisDevice();
 *     gpuDelegate = new GpuDelegate(delegateOptions):
 *     options.addDelegate(gpuDelegate);
 *   }
 * }
 * Interpreter interpreter = new Interpreter(modelBuffer, options);
 * }</pre>
 */
public class CompatibilityList implements Closeable {

  private static final long INVALID_COMPATIBILITY_LIST_HANDLE = 0;

  private long compatibilityListHandle = INVALID_COMPATIBILITY_LIST_HANDLE;

  /** Whether the GPU delegate is supported on this device. */
  public boolean isDelegateSupportedOnThisDevice() {
    if (compatibilityListHandle == INVALID_COMPATIBILITY_LIST_HANDLE) {
      throw new IllegalStateException("Trying to query a closed compatibilityList.");
    }
    return nativeIsDelegateSupportedOnThisDevice(compatibilityListHandle);
  }

  /** What options should be used for the GPU delegate. */
  public GpuDelegate.Options getBestOptionsForThisDevice() {
    // For forward compatibility, when the compatibilityList contains more information.
    return new GpuDelegate.Options();
  }

  public CompatibilityList() {
    GpuDelegateNative.init();
    compatibilityListHandle = createCompatibilityList();
  }

  /**
   * Frees TFLite resources in C runtime.
   *
   * <p>User is expected to call this method explicitly.
   */
  @Override
  public void close() {
    if (compatibilityListHandle != INVALID_COMPATIBILITY_LIST_HANDLE) {
      deleteCompatibilityList(compatibilityListHandle);
      compatibilityListHandle = INVALID_COMPATIBILITY_LIST_HANDLE;
    }
  }

  private static native long createCompatibilityList();

  private static native void deleteCompatibilityList(long handle);

  private static native boolean nativeIsDelegateSupportedOnThisDevice(long handle);
}
