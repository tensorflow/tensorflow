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
import org.tensorflow.lite.TensorFlowLite;

/** {@link Delegate} for NNAPI inference. */
public class NnApiDelegate implements Delegate, AutoCloseable {

  private static final long INVALID_DELEGATE_HANDLE = 0;

  private long delegateHandle;

  /** Delegate options. */
  public static final class Options {
    public Options() {}

    /**
     * undefined, specifies default behavior. so far, the default setting of NNAPI is
     * EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER
     */
    public static final int EXECUTION_PREFERENCE_UNDEFINED = -1;

    /**
     * Prefer executing in a way that minimizes battery drain. This is desirable for compilations
     * that will be executed often.
     */
    public static final int EXECUTION_PREFERENCE_LOW_POWER = 0;

    /**
     * Prefer returning a single answer as fast as possible, even if this causes more power
     * consumption.
     */
    public static final int EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER = 1;

    /**
     * Prefer maximizing the throughput of successive frames, for example when processing successive
     * frames coming from the camera.
     */
    public static final int EXECUTION_PREFERENCE_SUSTAINED_SPEED = 2;

    /**
     * Sets the inference preference for precision/compilation/runtime tradeoffs.
     *
     * @param preference One of EXECUTION_PREFERENCE_LOW_POWER,
     *     EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER, and EXECUTION_PREFERENCE_SUSTAINED_SPEED.
     */
    public Options setExecutionPreference(int preference) {
      this.executionPreference = preference;
      return this;
    }

    public Options setAcceleratorName(String name) {
      this.accelerator_name = name;
      return this;
    }

    public Options setCacheDir(String name) {
      this.cache_dir = name;
      return this;
    }

    public Options setModelToken(String name) {
      this.model_token = name;
      return this;
    }

    int executionPreference = EXECUTION_PREFERENCE_UNDEFINED;
    String accelerator_name = null;
    String cache_dir = null;
    String model_token = null;
  }

  public NnApiDelegate(Options options) {
    delegateHandle =
        createDelegate(
            options.executionPreference,
            options.accelerator_name,
            options.cache_dir,
            options.model_token);
  }

  public NnApiDelegate() {
    this(new Options());
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

  private static native long createDelegate(
      int preference, String device_name, String cache_dir, String model_token);

  private static native void deleteDelegate(long delegateHandle);

  static {
    // Ensure the native TensorFlow Lite libraries are available.
    TensorFlowLite.init();
  }
}
