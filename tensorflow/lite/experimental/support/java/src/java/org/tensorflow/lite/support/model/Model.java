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

package org.tensorflow.lite.support.model;

import android.content.Context;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.Map;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.SupportPreconditions;

/**
 * The wrapper class for a TFLite model and a TFLite interpreter.
 *
 * <p>Note: A {@link Model} can only holds 1 TFLite model at a time, and always holds a TFLite
 * interpreter instance to run it.
 */
public class Model {

  /** The runtime device type used for executing classification. */
  public enum Device {
    CPU,
    NNAPI,
    GPU
  }

  /**
   * Options for running the model. Configurable parameters includes:
   *
   * <ul>
   *   <li>{@code device} {@link Builder#setDevice(Device)} specifies the hardware to run the model.
   *       The default value is {@link Device#CPU}.
   *   <li>{@code numThreads} {@link Builder#setNumThreads(int)} specifies the number of threads
   *       used by TFLite inference. It's only effective when device is set to {@link Device#CPU}
   *       and default value is 1.
   * </ul>
   */
  public static class Options {
    private final Device device;
    private final int numThreads;

    /** Builder of {@link Options}. See its doc for details. */
    public static class Builder {
      private Device device = Device.CPU;
      private int numThreads = 1;

      public Builder setDevice(Device device) {
        this.device = device;
        return this;
      }

      public Builder setNumThreads(int numThreads) {
        this.numThreads = numThreads;
        return this;
      }

      public Options build() {
        return new Options(this);
      }
    }

    private Options(Builder builder) {
      device = builder.device;
      numThreads = builder.numThreads;
    }
  }

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  private final Interpreter interpreter;

  /** Path to tflite model file in asset folder. */
  private final String modelPath;

  /** The memory-mapped model data. */
  private final MappedByteBuffer byteModel;

  private final GpuDelegateProxy gpuDelegateProxy;

  /**
   * Builder for {@link Model}.
   *
   * @deprecated Please use {@link Model#createModel(Context, String, Options)}.
   */
  @Deprecated
  public static class Builder {
    private Device device = Device.CPU;
    private int numThreads = 1;
    private final String modelPath;
    private final MappedByteBuffer byteModel;

    /**
     * Creates a builder which loads tflite model from asset folder using memory-mapped files.
     *
     * @param context: Application context to access assets.
     * @param modelPath: Asset path of the model (.tflite file).
     * @throws IOException if an I/O error occurs when loading the tflite model.
     */
    @NonNull
    public Builder(@NonNull Context context, @NonNull String modelPath) throws IOException {
      this.modelPath = modelPath;
      byteModel = FileUtil.loadMappedFile(context, modelPath);
    }

    /** Sets running device. By default, TFLite will run on CPU. */
    @NonNull
    public Builder setDevice(Device device) {
      this.device = device;
      return this;
    }

    /** Sets number of threads. By default it's 1. */
    @NonNull
    public Builder setNumThreads(int numThreads) {
      this.numThreads = numThreads;
      return this;
    }

    // Note: The implementation is copied from `Model#createModel`. As the builder is going to be
    // deprecated, this function is also to be removed.
    @NonNull
    public Model build() {
      Options options = new Options.Builder().setNumThreads(numThreads).setDevice(device).build();
      return createModel(byteModel, modelPath, options);
    }
  }

  /**
   * Loads a model from assets and initialize TFLite interpreter.
   *
   * <p>The default options are: (1) CPU device; (2) one thread.
   *
   * @param context The App Context.
   * @param modelPath The path of the model file.
   * @throws IOException if any exception occurs when open the model file.
   */
  public static Model createModel(@NonNull Context context, @NonNull String modelPath)
      throws IOException {
    return createModel(context, modelPath, new Options.Builder().build());
  }

  /**
   * Loads a model from assets and initialize TFLite interpreter with given options.
   *
   * @see Options for details.
   * @param context The App Context.
   * @param modelPath The path of the model file.
   * @param options The options for running the model.
   * @throws IOException if any exception occurs when open the model file.
   */
  public static Model createModel(
      @NonNull Context context, @NonNull String modelPath, @NonNull Options options)
      throws IOException {
    SupportPreconditions.checkNotEmpty(
        modelPath, "Model path in the asset folder cannot be empty.");
    MappedByteBuffer byteModel = FileUtil.loadMappedFile(context, modelPath);
    return createModel(byteModel, modelPath, options);
  }

  /**
   * Creates a model with loaded {@link MappedByteBuffer}.
   *
   * @see Options for details.
   * @param byteModel The loaded TFLite model.
   * @param modelPath The original path of the model. It can be fetched later by {@link
   *     Model#getPath()}.
   * @param options The options for running the model.
   * @throws IllegalArgumentException if {@code options.device} is {@link Device#GPU} but
   *     "tensorflow-lite-gpu" is not linked to the project.
   */
  public static Model createModel(
      @NonNull MappedByteBuffer byteModel, @NonNull String modelPath, @NonNull Options options) {
    Interpreter.Options interpreterOptions = new Interpreter.Options();
    GpuDelegateProxy gpuDelegateProxy = null;
    switch (options.device) {
      case NNAPI:
        interpreterOptions.setUseNNAPI(true);
        break;
      case GPU:
        gpuDelegateProxy = GpuDelegateProxy.maybeNewInstance();
        SupportPreconditions.checkArgument(
            gpuDelegateProxy != null,
            "Cannot inference with GPU. Did you add \"tensorflow-lite-gpu\" as dependency?");
        interpreterOptions.addDelegate(gpuDelegateProxy);
        break;
      case CPU:
        break;
    }
    interpreterOptions.setNumThreads(options.numThreads);
    Interpreter interpreter = new Interpreter(byteModel, interpreterOptions);
    return new Model(modelPath, byteModel, interpreter, gpuDelegateProxy);
  }

  /** Returns the memory-mapped model data. */
  @NonNull
  public MappedByteBuffer getData() {
    return byteModel;
  }

  /** Returns the path of the model file stored in Assets. */
  @NonNull
  public String getPath() {
    return modelPath;
  }

  /**
   * Returns the output shape. Useful if output shape is only determined when graph is created.
   *
   * @throws IllegalStateException if the interpreter is closed.
   */
  public int[] getOutputTensorShape(int outputIndex) {
    return interpreter.getOutputTensor(outputIndex).shape();
  }

  /**
   * Runs model inference on multiple inputs, and returns multiple outputs.
   *
   * @param inputs an array of input data. The inputs should be in the same order as inputs of the
   *     model. Each input can be an array or multidimensional array, or a {@link
   *     java.nio.ByteBuffer} of primitive types including int, float, long, and byte. {@link
   *     java.nio.ByteBuffer} is the preferred way to pass large input data, whereas string types
   *     require using the (multi-dimensional) array input path. When {@link java.nio.ByteBuffer} is
   *     used, its content should remain unchanged until model inference is done.
   * @param outputs a map mapping output indices to multidimensional arrays of output data or {@link
   *     java.nio.ByteBuffer}s of primitive types including int, float, long, and byte. It only
   *     needs to keep entries for the outputs to be used.
   */
  public void run(@NonNull Object[] inputs, @NonNull Map<Integer, Object> outputs) {
    interpreter.runForMultipleInputsOutputs(inputs, outputs);
  }

  public void close() {
    if (interpreter != null) {
      interpreter.close();
    }
    if (gpuDelegateProxy != null) {
      gpuDelegateProxy.close();
    }
  }

  private Model(
      @NonNull String modelPath,
      @NonNull MappedByteBuffer byteModel,
      @NonNull Interpreter interpreter,
      @Nullable GpuDelegateProxy gpuDelegateProxy) {
    this.modelPath = modelPath;
    this.byteModel = byteModel;
    this.interpreter = interpreter;
    this.gpuDelegateProxy = gpuDelegateProxy;
  }
}
