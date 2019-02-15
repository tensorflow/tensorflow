/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

package com.example.android.tflitecamerademo;

import org.tensorflow.lite.Delegate;

/**
 * Helper class for {@code GpuDelegate}.
 *
 * <p>WARNING: This is an experimental API and subject to change.
 */
public class GpuDelegateHelper {
  private GpuDelegateHelper() {}

  /** Checks whether {@code GpuDelegate} is available. */
  public static boolean isGpuDelegateAvailable() {
    try {
      Class.forName("org.tensorflow.lite.experimental.GpuDelegate");
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  /** Returns an instance of {@code GpuDelegate} if available. */
  public static Delegate createGpuDelegate() {
    try {
      return Class.forName("org.tensorflow.lite.experimental.GpuDelegate")
          .asSubclass(Delegate.class)
          .getDeclaredConstructor()
          .newInstance();
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }
}
