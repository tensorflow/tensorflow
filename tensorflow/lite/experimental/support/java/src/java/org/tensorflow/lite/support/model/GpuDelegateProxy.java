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

package org.tensorflow.lite.support.model;

import android.util.Log;
import java.io.Closeable;
import java.io.IOException;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.tensorflow.lite.Delegate;

/**
 * Helper class to create and call necessary methods of {@code GpuDelegate} which is not a strict
 * dependency.
 */
class GpuDelegateProxy implements Delegate, Closeable {

  private static final String TAG = "GpuDelegateProxy";

  private final Delegate proxiedDelegate;
  private final Closeable proxiedCloseable;

  @Nullable
  public static GpuDelegateProxy maybeNewInstance() {
    try {
      Class<?> clazz = Class.forName("org.tensorflow.lite.gpu.GpuDelegate");
      Object instance = clazz.getDeclaredConstructor().newInstance();
      return new GpuDelegateProxy(instance);
    } catch (ReflectiveOperationException e) {
      Log.e(TAG, "Failed to create the GpuDelegate dynamically.", e);
      return null;
    }
  }

  /** Calls {@code close()} method of the delegate. */
  @Override
  public void close() {
    try {
      proxiedCloseable.close();
    } catch (IOException e) {
      // Should not trigger, because GpuDelegate#close never throws. The catch is required because
      // of Closeable#close.
      Log.e(TAG, "Failed to close the GpuDelegate.", e);
    }
  }

  /** Calls {@code getNativeHandle()} method of the delegate. */
  @Override
  public long getNativeHandle() {
    return proxiedDelegate.getNativeHandle();
  }

  private GpuDelegateProxy(Object instance) {
    this.proxiedCloseable = (Closeable) instance;
    this.proxiedDelegate = (Delegate) instance;
  }
}
