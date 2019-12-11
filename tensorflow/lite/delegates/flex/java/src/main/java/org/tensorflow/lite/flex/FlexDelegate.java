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

package org.tensorflow.lite.flex;

import java.io.Closeable;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.annotations.UsedByReflection;

/** {@link Delegate} for using select TensorFlow ops. */
@UsedByReflection("Interpreter")
public class FlexDelegate implements Delegate, Closeable {

  private static final long INVALID_DELEGATE_HANDLE = 0;
  private static final String TFLITE_FLEX_LIB = "tensorflowlite_flex_jni";

  private long delegateHandle;

  @UsedByReflection("Interpreter")
  public FlexDelegate() {
    delegateHandle = nativeCreateDelegate();
  }

  @Override
  @UsedByReflection("Interpreter")
  public long getNativeHandle() {
    return delegateHandle;
  }

  /**
   * Releases native resources held by the delegate.
   *
   * <p>User is expected to call this method explicitly.
   */
  @Override
  @UsedByReflection("Interpreter")
  public void close() {
    if (delegateHandle != INVALID_DELEGATE_HANDLE) {
      nativeDeleteDelegate(delegateHandle);
      delegateHandle = INVALID_DELEGATE_HANDLE;
    }
  }

  public static void initTensorFlowForTesting() {
    nativeInitTensorFlow();
  }

  static {
    System.loadLibrary(TFLITE_FLEX_LIB);
  }

  private static native long nativeInitTensorFlow();

  private static native long nativeCreateDelegate();

  private static native void nativeDeleteDelegate(long delegateHandle);
}
