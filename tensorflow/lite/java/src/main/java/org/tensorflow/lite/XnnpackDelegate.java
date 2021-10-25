/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite;

/** Wrapper for a native TensorFlow Lite XNNPACK Delegate. */
class XnnpackDelegate implements Delegate, AutoCloseable {
  XnnpackDelegate(long nativeHandle, long deleteFunction) {
    this.nativeHandle = nativeHandle;
    this.deleteFunction = deleteFunction;
  }

  @Override
  public long getNativeHandle() {
    return nativeHandle;
  }

  @Override
  public void close() {
    applyDeleteFunction(deleteFunction, nativeHandle);
  }

  // Apply deleteFunction to nativeHandle.
  private static native void applyDeleteFunction(long deleteFunction, long nativeHandle);

  private long nativeHandle; // C/C++ type: "TFLiteDelegate *", i.e. pointer to TFLiteDelegate.
  private long deleteFunction; // C/C++ type: "void (*)(TFLiteDelegate*)": pointer to function.
}
