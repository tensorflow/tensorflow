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

package org.tensorflow.contrib.android;

/**
 * Accumulate and analyze stats from metadata obtained from Session.Runner.run.
 */
public class RunStats implements AutoCloseable {

  /**
   * Options to be provided to a {@link org.tensorflow.Session.Runner} to enable stats accumulation.
   */
  public static byte[] getRunOptions() {
    return FULL_TRACE_RUN_OPTIONS;
  }

  public RunStats() {
    nativeHandle = allocate();
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      delete(nativeHandle);
    }
    nativeHandle = 0;
  }

  /**
   * Accumulate stats obtained when executing a graph.
   */
  public synchronized void add(byte[] runMetadata) {
    add(nativeHandle, runMetadata);
  }

  /**
   * Summary of the accumulated runtime stats.
   */
  public synchronized String summary() {
    return summary(nativeHandle);
  }

  private long nativeHandle;

  // Hack: This is what a serialized RunOptions protocol buffer with trace_level: FULL_TRACE ends up as.
  private static final byte[] FULL_TRACE_RUN_OPTIONS = {0x08, 0x03};

  private static native long allocate();

  private static native void delete(long handle);

  private static native void add(long handle, byte[] runMetadata);

  private static native String summary(long handle);
}
