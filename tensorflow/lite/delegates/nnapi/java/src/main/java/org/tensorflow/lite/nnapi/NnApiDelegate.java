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
import org.tensorflow.lite.InterpreterFactoryApi;
import org.tensorflow.lite.TensorFlowLite;

/** {@link Delegate} for NNAPI inference. */
public class NnApiDelegate implements Delegate, AutoCloseable {
  /**
   * Interface used for implementing NnApiDelegate. This is for use only by packages in
   * org.tensorflow.lite.
   *
   * @hide
   */
  public static interface PrivateInterface extends Delegate, AutoCloseable {
    /** See NnApiDelegate#getNnApiErrno. */
    public int getNnapiErrno();

    @Override
    public void close();
  }

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
     *     EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER, or EXECUTION_PREFERENCE_SUSTAINED_SPEED.
     */
    public Options setExecutionPreference(int preference) {
      this.executionPreference = preference;
      return this;
    }

    /**
     * Specifies the name of the target accelerator to be used by NNAPI. If this parameter is
     * specified the {@link #setUseNnapiCpu(boolean)} method won't have any effect.
     *
     * <p>Only effective on Android 10 (API level 29) and above.
     */
    public Options setAcceleratorName(String name) {
      this.acceleratorName = name;
      return this;
    }

    /**
     * Configure the location to be used to store model compilation cache entries. If either {@code
     * cacheDir} or {@code modelToken} parameters are unset NNAPI caching will be disabled.
     *
     * <p>Only effective on Android 10 (API level 29) and above.
     */
    public Options setCacheDir(String cacheDir) {
      this.cacheDir = cacheDir;
      return this;
    }

    /**
     * Sets the token to be used to identify this model in the model compilation cache. If either
     * {@code cacheDir} or {@code modelToken} parameters are unset NNAPI caching will be disabled.
     *
     * <p>Only effective on Android 10 (API level 29) and above.
     */
    public Options setModelToken(String modelToken) {
      this.modelToken = modelToken;
      return this;
    }

    /**
     * Sets the maximum number of graph partitions that the delegate will try to delegate. If more
     * partitions could be delegated than the limit, the ones with the larger number of nodes will
     * be chosen. If unset it will use the NNAPI default limit.
     */
    public Options setMaxNumberOfDelegatedPartitions(int limit) {
      this.maxDelegatedPartitions = limit;
      return this;
    }

    /**
     * Enable or disable the NNAPI CPU Device "nnapi-reference". If unset it will use the NNAPI
     * default settings.
     *
     * <p>Only effective on Android 10 (API level 29) and above.
     */
    public Options setUseNnapiCpu(boolean enable) {
      this.useNnapiCpu = enable;
      return this;
    }

    /**
     * Enable or disable to allow fp32 computation to be run in fp16 in NNAPI. See
     * https://source.android.com/devices/neural-networks#android-9
     *
     * <p>Only effective on Android 9 (API level 28) and above.
     */
    public Options setAllowFp16(boolean enable) {
      this.allowFp16 = enable;
      return this;
    }

    public Options setNnApiSupportLibraryHandle(long handle) {
      this.nnApiSupportLibraryHandle = handle;
      return this;
    }

    /**
     * Returns the inference preference for precision/compilation/runtime tradeoffs.
     *
     * @return One of EXECUTION_PREFERENCE_LOW_POWER, EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER, or
     *     EXECUTION_PREFERENCE_SUSTAINED_SPEED.
     */
    public int getExecutionPreference() {
      return this.executionPreference;
    }

    /** Returns the name of the target accelerator to be used by NNAPI. */
    public String getAcceleratorName() {
      return this.acceleratorName;
    }

    /** Returns the location to be used to store model compilation cache entries. */
    public String getCacheDir() {
      return this.cacheDir;
    }

    /** Returns the token to be used to identify this model in the model compilation cache. */
    public String getModelToken() {
      return this.modelToken;
    }

    /**
     * Returns the maximum number of graph partitions that the delegate will try to delegate. Or -1
     * if no maximum has neen set.
     */
    public int getMaxNumberOfDelegatedPartitions() {
      return this.maxDelegatedPartitions == null ? -1 : this.maxDelegatedPartitions;
    }

    /**
     * Returns whether the NNAPI CPU Device "nnapi-reference" is enabled.
     *
     * @return true if the NNAPI CPU Device "nnapi-reference" is explicitly enabled.
     * @return false if the NNAPI CPU Device "nnapi-reference" is explicitly disabled.
     * @return null if use of the NNAPI CPU Device "nnapi-reference" will be determined by the NNAPI
     *     default settings.
     */
    public Boolean getUseNnapiCpu() {
      return this.useNnapiCpu;
    }

    /** Returns true if allowing fp32 computation to be run in fp16 in NNAPI is enabled. */
    public boolean getAllowFp16() {
      return this.allowFp16 != null && this.allowFp16;
    }

    /** Returns the native handle to the NNAPI Support Library. */
    public long getNnApiSupportLibraryHandle() {
      return this.nnApiSupportLibraryHandle;
    }

    private int executionPreference = EXECUTION_PREFERENCE_UNDEFINED;
    private String acceleratorName = null;
    private String cacheDir = null;
    private String modelToken = null;
    private Integer maxDelegatedPartitions = null;
    private Boolean useNnapiCpu = null;
    private Boolean allowFp16 = null;
    private long nnApiSupportLibraryHandle = 0;
  }

  private Options options;
  private PrivateInterface impl;
  private boolean initialized;

  public NnApiDelegate(Options options) {
    // Ensure the native TensorFlow Lite libraries are available.
    TensorFlowLite.init();
    this.options = options;
    // this.impl will be initialized later.
  }

  public NnApiDelegate() {
    this(new Options());
  }

  /**
   * Called when an interpreter is constructed with this delegate. This method is only for use by TF
   * Lite itself, and should not be used by application developers.
   *
   * @hide
   */
  public void initWithInterpreterFactoryApi(InterpreterFactoryApi interpreterFactoryApi) {
    impl = interpreterFactoryApi.createNnApiDelegateImpl(options);
    initialized = true;
  }

  @Override
  public long getNativeHandle() {
    checkNotClosed();
    return impl.getNativeHandle();
  }

  /**
   * Frees TFLite resources in C runtime.
   *
   * <p>User is expected to call this method explicitly.
   */
  @Override
  public void close() {
    if (impl != null) {
      impl.close();
      impl = null;
    }
  }

  /**
   * Returns the latest error code returned by an NNAPI call or zero if NO calls to NNAPI failed.
   * The error code is reset when the delegate is associated with an <a
   * href=https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Interpreter>interpreter</a>.
   *
   * <p>For details on NNAPI error codes see <a
   * href="https://developer.android.com/ndk/reference/group/neural-networks#resultcode">the NNAPI
   * documentation</a>.
   *
   * @throws IllegalStateException if the method is called after {@link #close() close}.
   */
  public int getNnapiErrno() {
    if (!initialized) {
      return 0 /*ANEURALNETWORKS_NO_ERROR*/;
    }
    checkNotClosed();
    return impl.getNnapiErrno();
  }

  /**
   * Returns true if any NNAPI call failed since this delegate was associated with an <a
   * href=https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Interpreter>interpreter</a>.
   *
   * @throws IllegalStateException if the method is called after {@link #close() close}.
   */
  public boolean hasErrors() {
    return getNnapiErrno() != 0 /*ANEURALNETWORKS_NO_ERROR*/;
  }

  private void checkNotClosed() {
    if (impl == null) {
      throw new IllegalStateException(
          initialized
              ? "Should not access delegate after delegate has been closed."
              : "Should not access delegate before interpreter has been constructed.");
    }
  }
}
