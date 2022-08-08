/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite.gpu;

import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.DelegateFactory;
import org.tensorflow.lite.RuntimeFlavor;

/** {@link DelegateFactory} for creating a {@link GpuDelegate}. */
public class GpuDelegateFactory implements DelegateFactory {

  private static final String GPU_DELEGATE_CLASS_NAME = "GpuDelegate";

  private final Options options;

  /** Delegate options. */
  public static class Options {
    public Options() {}

    /**
     * Delegate will be used only once, therefore, bootstrap/init time should be taken into account.
     */
    public static final int INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER = 0;

    /**
     * Prefer maximizing the throughput. Same delegate will be used repeatedly on multiple inputs.
     */
    public static final int INFERENCE_PREFERENCE_SUSTAINED_SPEED = 1;

    /**
     * Sets whether precision loss is allowed.
     *
     * @param precisionLossAllowed When `true` (default), the GPU may quantify tensors, downcast
     *     values, process in FP16. When `false`, computations are carried out in 32-bit floating
     *     point.
     */
    @CanIgnoreReturnValue
    public Options setPrecisionLossAllowed(boolean precisionLossAllowed) {
      this.precisionLossAllowed = precisionLossAllowed;
      return this;
    }

    /**
     * Enables running quantized models with the delegate.
     *
     * <p>WARNING: This is an experimental API and subject to change.
     *
     * @param quantizedModelsAllowed When {@code true} (default), the GPU may run quantized models.
     */
    @CanIgnoreReturnValue
    public Options setQuantizedModelsAllowed(boolean quantizedModelsAllowed) {
      this.quantizedModelsAllowed = quantizedModelsAllowed;
      return this;
    }

    /**
     * Sets the inference preference for precision/compilation/runtime tradeoffs.
     *
     * @param preference One of `INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER` (default),
     *     `INFERENCE_PREFERENCE_SUSTAINED_SPEED`.
     */
    @CanIgnoreReturnValue
    public Options setInferencePreference(int preference) {
      this.inferencePreference = preference;
      return this;
    }

    /**
     * Enables serialization on the delegate. Note non-null {@code serializationDir} and {@code
     * modelToken} are required for serialization.
     *
     * <p>WARNING: This is an experimental API and subject to change.
     *
     * @param serializationDir The directory to use for storing data. Caller is responsible to
     *     ensure the model is not stored in a public directory. It's recommended to use {@link
     *     android.content.Context#getCodeCacheDir()} to provide a private location for the
     *     application on Android.
     * @param modelToken The token to be used to identify the model. Caller is responsible to ensure
     *     the token is unique to the model graph and data.
     */
    @CanIgnoreReturnValue
    public Options setSerializationParams(String serializationDir, String modelToken) {
      this.serializationDir = serializationDir;
      this.modelToken = modelToken;
      return this;
    }

    public boolean isPrecisionLossAllowed() {
      return precisionLossAllowed;
    }

    public boolean areQuantizedModelsAllowed() {
      return quantizedModelsAllowed;
    }

    public int getInferencePreference() {
      return inferencePreference;
    }

    public String getSerializationDir() {
      return serializationDir;
    }

    public String getModelToken() {
      return modelToken;
    }

    private boolean precisionLossAllowed = true;
    boolean quantizedModelsAllowed = true;
    int inferencePreference = INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
    String serializationDir = null;
    String modelToken = null;
  }

  public GpuDelegateFactory() {
    this(new Options());
  }

  public GpuDelegateFactory(Options options) {
    this.options = options;
  }

  @Override
  public Delegate create(RuntimeFlavor runtimeFlavor) {
    String packageName;
    switch (runtimeFlavor) {
      case APPLICATION:
        packageName = "org.tensorflow.lite.gpu";
        break;
      case SYSTEM:
        packageName = "com.google.android.gms.tflite.gpu";
        break;
      default:
        throw new IllegalArgumentException("Unsupported runtime flavor " + runtimeFlavor);
    }
    try {
      Class<?> delegateClass = Class.forName(packageName + "." + GPU_DELEGATE_CLASS_NAME);
      Constructor<?> constructor = delegateClass.getDeclaredConstructor(Options.class);
      return (Delegate) constructor.newInstance(options);
    } catch (ClassNotFoundException
        | IllegalAccessException
        | InstantiationException
        | NoSuchMethodException
        | InvocationTargetException e) {
      throw new IllegalStateException("Error creating GPU delegate", e);
    }
  }
}
