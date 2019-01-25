/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

/**
 * Wrapper for a native TensorFlow Lite Delegate.
 *
 * <p>WARNING: This is an experimental interface that is subject to change.
 *
 * <p>If a delegate implementation holds additional resources or memory that should be explicitly
 * freed, then best practice is to add a {@code close()} method to the implementation and have the
 * client call that explicitly when the delegate instance is no longer in use. While this approach
 * technically allows sharing of a single delegate instance across multiple interpreter instances,
 * the delegate implementation must explicitly support this.
 */
public interface Delegate {
  /**
   * Returns a native handle to the TensorFlow Lite delegate implementation.
   *
   * <p>Note: The Java {@link Delegate} maintains ownership of the native delegate instance, and
   * must ensure its existence for the duration of usage with any {@link Interpreter}.
   *
   * @return The native delegate handle.
   */
  public long getNativeHandle();
}
