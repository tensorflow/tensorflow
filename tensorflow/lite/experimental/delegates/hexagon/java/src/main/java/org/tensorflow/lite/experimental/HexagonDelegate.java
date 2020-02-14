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

import android.content.Context;
import java.io.Closeable;
import org.tensorflow.lite.Delegate;

/** {@link Delegate} for Hexagon inference. */
public class HexagonDelegate implements Delegate, Closeable {

  private static final long INVALID_DELEGATE_HANDLE = 0;
  private static final String TFLITE_HEXAGON_LIB = "tensorflowlite_hexagon_jni";
  private static volatile boolean nativeLibraryLoaded = false;

  private long delegateHandle;

  /*
   * Creates a new HexagonDelegate object given the current 'context'.
   * Throws UnsupportedOperationException if Hexagon DSP delegation is not available
   * on this device.
   */
  public HexagonDelegate(Context context) throws UnsupportedOperationException {
    ensureNativeLibraryLoaded();
    setAdspLibraryPath(context.getApplicationInfo().nativeLibraryDir);
    delegateHandle = createDelegate();
    if (delegateHandle == INVALID_DELEGATE_HANDLE) {
      throw new UnsupportedOperationException("This Device doesn't support Hexagon DSP execution.");
    }
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

  private static void ensureNativeLibraryLoaded() {
    if (nativeLibraryLoaded) {
      return;
    }
    try {
      System.loadLibrary(TFLITE_HEXAGON_LIB);
      nativeLibraryLoaded = true;
    } catch (Exception e) {
      throw new UnsupportedOperationException("Failed to load native Hexagon shared library: " + e);
    }
  }

  private static native long createDelegate();

  private static native void deleteDelegate(long delegateHandle);

  private static native boolean setAdspLibraryPath(String libraryPath);
}
