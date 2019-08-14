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

package org.tensorflow.lite.nnapi;

import org.tensorflow.lite.Delegate;

/** {@link Delegate} for NNAPI inference. */
public class NnApiDelegate implements Delegate, AutoCloseable {

  private static final long INVALID_DELEGATE_HANDLE = 0;

  private long delegateHandle;

  public NnApiDelegate() {
    delegateHandle = createDelegate();
  }

  @Override
  public long getNativeHandle() {
    return delegateHandle;
  }

  /**
   * The NNAPI delegate is singleton. Nothing to delete for now, so mark the handle invalid only.
   */
  @Override
  public void close() {
    if (delegateHandle != INVALID_DELEGATE_HANDLE) {
      delegateHandle = INVALID_DELEGATE_HANDLE;
    }
  }

  private static native long createDelegate();
}
