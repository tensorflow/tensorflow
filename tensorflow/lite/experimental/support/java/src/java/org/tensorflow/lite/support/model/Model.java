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
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.SupportPrecondtions;

/** Class to load tflite models from App asset folder or remote sources. */
public class Model {

  /** An instance of the driver c /** The runtime device type used for executing classification. */
  public enum Device {
    CPU,
    NNAPI,
    GPU
  }

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  private final Interpreter interpreter;

  /** Path to tflite model file in asset folder. */
  private final String modelPath;

  /** The memory-mapped model data. */
  private final MappedByteBuffer byteModel;

  private final GpuDelegate gpuDelegate;

  /** Builder for {@link Model}. */
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

    @NonNull
    public Model build() {
      return new Model(modelPath, byteModel, device, numThreads);
    }
  }

  /** Return the memory-mapped model data. */
  @NonNull
  public MappedByteBuffer getData() {
    return byteModel;
  }

  /** Return the path of the model file stored in Assets. */
  @NonNull
  public String getPath() {
    return modelPath;
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
    if (gpuDelegate != null) {
      gpuDelegate.close();
    }
  }

  private Model(
      @NonNull String modelPath,
      @NonNull MappedByteBuffer byteModel,
      Device device,
      int numThreads) {
    SupportPrecondtions.checkNotNull(byteModel, "Model file cannot be null.");
    SupportPrecondtions.checkNotEmpty(modelPath, "Model path in the asset folder cannot be empty.");

    this.modelPath = modelPath;
    this.byteModel = byteModel;
    Interpreter.Options interpreterOptions = new Interpreter.Options();
    gpuDelegate = device == Device.GPU ? new GpuDelegate() : null;
    switch (device) {
      case NNAPI:
        interpreterOptions.setUseNNAPI(true);
        break;
      case GPU:
        interpreterOptions.addDelegate(gpuDelegate);
        break;
      case CPU:
        break;
    }
    interpreterOptions.setNumThreads(numThreads);
    interpreter = new Interpreter(byteModel, interpreterOptions);
  }
}
