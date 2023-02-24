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

import java.io.File;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;
import org.tensorflow.lite.acceleration.ValidatedAccelerationConfig;
import org.tensorflow.lite.nnapi.NnApiDelegate;

/**
 * Interface to TensorFlow Lite model interpreter, excluding experimental methods.
 *
 * <p>An {@code InterpreterApi} instance encapsulates a pre-trained TensorFlow Lite model, in which
 * operations are executed for model inference.
 *
 * <p>For example, if a model takes only one input and returns only one output:
 *
 * <pre> {@code
 * try (InterpreterApi interpreter =
 *     new InterpreterApi.create(file_of_a_tensorflowlite_model)) {
 *   interpreter.run(input, output);
 * }}</pre>
 *
 * <p>If a model takes multiple inputs or outputs:
 *
 * <pre> {@code
 * Object[] inputs = {input0, input1, ...};
 * Map<Integer, Object> map_of_indices_to_outputs = new HashMap<>();
 * FloatBuffer ith_output = FloatBuffer.allocateDirect(3 * 2 * 4);  // Float tensor, shape 3x2x4.
 * ith_output.order(ByteOrder.nativeOrder());
 * map_of_indices_to_outputs.put(i, ith_output);
 * try (InterpreterApi interpreter =
 *     new InterpreterApi.create(file_of_a_tensorflowlite_model)) {
 *   interpreter.runForMultipleInputsOutputs(inputs, map_of_indices_to_outputs);
 * }}</pre>
 *
 * <p>If a model takes or produces string tensors:
 *
 * <pre> {@code
 * String[] input = {"foo", "bar"};  // Input tensor shape is [2].
 * String[] output = new String[3][2];  // Output tensor shape is [3, 2].
 * try (InterpreterApi interpreter =
 *     new InterpreterApi.create(file_of_a_tensorflowlite_model)) {
 *   interpreter.runForMultipleInputsOutputs(input, output);
 * }}</pre>
 *
 * <p>Orders of inputs and outputs are determined when converting TensorFlow model to TensorFlowLite
 * model with Toco, as are the default shapes of the inputs.
 *
 * <p>When inputs are provided as (multi-dimensional) arrays, the corresponding input tensor(s) will
 * be implicitly resized according to that array's shape. When inputs are provided as {@link
 * java.nio.Buffer} types, no implicit resizing is done; the caller must ensure that the {@link
 * java.nio.Buffer} byte size either matches that of the corresponding tensor, or that they first
 * resize the tensor via {@link #resizeInput(int, int[])}. Tensor shape and type information can be
 * obtained via the {@link Tensor} class, available via {@link #getInputTensor(int)} and {@link
 * #getOutputTensor(int)}.
 *
 * <p><b>WARNING:</b>{@code InterpreterApi} instances are <b>not</b> thread-safe.
 *
 * <p><b>WARNING:</b>An {@code InterpreterApi} instance owns resources that <b>must</b> be
 * explicitly freed by invoking {@link #close()}
 *
 * <p>The TFLite library is built against NDK API 19. It may work for Android API levels below 19,
 * but is not guaranteed.
 */
public interface InterpreterApi extends AutoCloseable {

  /** An options class for controlling runtime interpreter behavior. */
  class Options {

    public Options() {
      this.delegates = new ArrayList<>();
      this.delegateFactories = new ArrayList<>();
    }

    public Options(Options other) {
      this.numThreads = other.numThreads;
      this.useNNAPI = other.useNNAPI;
      this.allowCancellation = other.allowCancellation;
      this.delegates = new ArrayList<>(other.delegates);
      this.delegateFactories = new ArrayList<>(other.delegateFactories);
      this.runtime = other.runtime;
      this.validatedAccelerationConfig = other.validatedAccelerationConfig;
      this.useXNNPACK = other.useXNNPACK;
    }

    /**
     * Sets the number of threads to be used for ops that support multi-threading.
     *
     * <p>{@code numThreads} should be {@code >= -1}. Setting {@code numThreads} to 0 has the effect
     * of disabling multithreading, which is equivalent to setting {@code numThreads} to 1. If
     * unspecified, or set to the value -1, the number of threads used will be
     * implementation-defined and platform-dependent.
     */
    public Options setNumThreads(int numThreads) {
      this.numThreads = numThreads;
      return this;
    }

    /**
     * Returns the number of threads to be used for ops that support multi-threading.
     *
     * <p>{@code numThreads} should be {@code >= -1}. Values of 0 (or 1) disable multithreading.
     * Default value is -1: the number of threads used will be implementation-defined and
     * platform-dependent.
     */
    public int getNumThreads() {
      return numThreads;
    }

    /** Sets whether to use NN API (if available) for op execution. Defaults to false (disabled). */
    public Options setUseNNAPI(boolean useNNAPI) {
      this.useNNAPI = useNNAPI;
      return this;
    }

    /**
     * Returns whether to use NN API (if available) for op execution. Default value is false
     * (disabled).
     */
    public boolean getUseNNAPI() {
      return useNNAPI != null && useNNAPI;
    }

    /**
     * Advanced: Set if the interpreter is able to be cancelled.
     *
     * <p>Interpreters may have an experimental API <a
     * href="https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Interpreter#setCancelled(boolean)">setCancelled(boolean)</a>.
     * If this interpreter is cancellable and such a method is invoked, a cancellation flag will be
     * set to true. The interpreter will check the flag between Op invocations, and if it's {@code
     * true}, the interpreter will stop execution. The interpreter will remain a cancelled state
     * until explicitly "uncancelled" by {@code setCancelled(false)}.
     */
    public Options setCancellable(boolean allow) {
      this.allowCancellation = allow;
      return this;
    }

    /**
     * Advanced: Returns whether the interpreter is able to be cancelled.
     *
     * <p>Interpreters may have an experimental API <a
     * href="https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Interpreter#setCancelled(boolean)">setCancelled(boolean)</a>.
     * If this interpreter is cancellable and such a method is invoked, a cancellation flag will be
     * set to true. The interpreter will check the flag between Op invocations, and if it's {@code
     * true}, the interpreter will stop execution. The interpreter will remain a cancelled state
     * until explicitly "uncancelled" by {@code setCancelled(false)}.
     */
    public boolean isCancellable() {
      return allowCancellation != null && allowCancellation;
    }

    /**
     * Adds a {@link Delegate} to be applied during interpreter creation.
     *
     * <p>Delegates added here are applied before any delegates created from a {@link
     * DelegateFactory} that was added with {@link #addDelegateFactory}.
     *
     * <p>Note that TF Lite in Google Play Services (see {@link #setRuntime}) does not support
     * external (developer-provided) delegates, and adding a {@link Delegate} other than {@link
     * NnApiDelegate} here is not allowed when using TF Lite in Google Play Services.
     */
    public Options addDelegate(Delegate delegate) {
      delegates.add(delegate);
      return this;
    }

    /**
     * Returns the list of delegates intended to be applied during interpreter creation that have
     * been registered via {@code addDelegate}.
     */
    public List<Delegate> getDelegates() {
      return Collections.unmodifiableList(delegates);
    }

    /**
     * Adds a {@link DelegateFactory} which will be invoked to apply its created {@link Delegate}
     * during interpreter creation.
     *
     * <p>Delegates from a delegated factory that was added here are applied after any delegates
     * added with {@link #addDelegate}.
     */
    public Options addDelegateFactory(DelegateFactory delegateFactory) {
      delegateFactories.add(delegateFactory);
      return this;
    }

    /**
     * Returns the list of delegate factories that have been registered via {@code
     * addDelegateFactory}).
     */
    public List<DelegateFactory> getDelegateFactories() {
      return Collections.unmodifiableList(delegateFactories);
    }

    /**
     * Enum to represent where to get the TensorFlow Lite runtime implementation from.
     *
     * <p>The difference between this class and the RuntimeFlavor class: This class specifies a
     * <em>preference</em> which runtime to use, whereas {@link RuntimeFlavor} specifies which exact
     * runtime <em>is</em> being used.
     */
    public enum TfLiteRuntime {
      /**
       * Use a TF Lite runtime implementation that is linked into the application. If there is no
       * suitable TF Lite runtime implementation linked into the application, then attempting to
       * create an InterpreterApi instance with this TfLiteRuntime setting will throw an
       * IllegalStateException exception (even if the OS or system services could provide a TF Lite
       * runtime implementation).
       *
       * <p>This is the default setting. This setting is also appropriate for apps that must run on
       * systems that don't provide a TF Lite runtime implementation.
       */
      FROM_APPLICATION_ONLY,

      /**
       * Use a TF Lite runtime implementation provided by the OS or system services. This will be
       * obtained from a system library / shared object / service, such as Google Play Services. It
       * may be newer than the version linked into the application (if any). If there is no suitable
       * TF Lite runtime implementation provided by the system, then attempting to create an
       * InterpreterApi instance with this TfLiteRuntime setting will throw an IllegalStateException
       * exception (even if there is a TF Lite runtime implementation linked into the application).
       *
       * <p>This setting is appropriate for code that will use a system-provided TF Lite runtime,
       * which can reduce app binary size and can be updated more frequently.
       */
      FROM_SYSTEM_ONLY,

      /**
       * Use a system-provided TF Lite runtime implementation, if any, otherwise use the TF Lite
       * runtime implementation linked into the application, if any. If no suitable TF Lite runtime
       * can be found in any location, then attempting to create an InterpreterApi instance with
       * this TFLiteRuntime setting will throw an IllegalStateException. If there is both a suitable
       * TF Lite runtime linked into the application and also a suitable TF Lite runtime provided by
       * the system, the one provided by the system will be used.
       *
       * <p>This setting is suitable for use in code that doesn't care where the TF Lite runtime is
       * coming from (e.g. middleware layers).
       */
      PREFER_SYSTEM_OVER_APPLICATION,
    }

    /** Specify where to get the TF Lite runtime implementation from. */
    public Options setRuntime(TfLiteRuntime runtime) {
      this.runtime = runtime;
      return this;
    }

    /** Return where to get the TF Lite runtime implementation from. */
    public TfLiteRuntime getRuntime() {
      return runtime;
    }

    /** Specify the acceleration configuration. */
    public Options setAccelerationConfig(ValidatedAccelerationConfig config) {
      this.validatedAccelerationConfig = config;
      return this;
    }

    /** Return the acceleration configuration. */
    public ValidatedAccelerationConfig getAccelerationConfig() {
      return this.validatedAccelerationConfig;
    }

    /**
     * Enable or disable an optimized set of CPU kernels (provided by XNNPACK). Enabled by default.
     */
    public Options setUseXNNPACK(boolean useXNNPACK) {
      this.useXNNPACK = useXNNPACK;
      return this;
    }

    public boolean getUseXNNPACK() {
      // A null value indicates the default behavior, which is currently to apply the delegate.
      return useXNNPACK == null || useXNNPACK.booleanValue();
    }

    TfLiteRuntime runtime = TfLiteRuntime.FROM_APPLICATION_ONLY;
    int numThreads = -1;
    Boolean useNNAPI;

    /**
     * Note: the initial "null" value indicates default behavior (XNNPACK delegate will be applied
     * by default whenever possible).
     *
     * <p>Disabling this flag will disable use of a highly optimized set of CPU kernels provided via
     * the XNNPACK delegate. Currently, this is restricted to a subset of floating point operations.
     * See
     * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/xnnpack/README.md
     * for more details.
     */
    Boolean useXNNPACK;

    Boolean allowCancellation;
    ValidatedAccelerationConfig validatedAccelerationConfig;

    // See InterpreterApi.Options#addDelegate.
    final List<Delegate> delegates;
    // See InterpreterApi.Options#addDelegateFactory.
    private final List<DelegateFactory> delegateFactories;
  }

  /**
   * Constructs an {@link InterpreterApi} instance, using the specified model and options. The model
   * will be loaded from a file.
   *
   * @param modelFile A file containing a pre-trained TF Lite model.
   * @param options A set of options for customizing interpreter behavior.
   * @throws IllegalArgumentException if {@code modelFile} does not encode a valid TensorFlow Lite
   *     model.
   */
  @SuppressWarnings("StaticOrDefaultInterfaceMethod")
  static InterpreterApi create(@NonNull File modelFile, InterpreterApi.Options options) {
    TfLiteRuntime runtime = (options == null ? null : options.getRuntime());
    InterpreterFactoryApi factory = TensorFlowLite.getFactory(runtime);
    return factory.create(modelFile, options);
  }

  /**
   * Constructs an {@link InterpreterApi} instance, using the specified model and options. The model
   * will be read from a {@code ByteBuffer}.
   *
   * @param byteBuffer A pre-trained TF Lite model, in binary serialized form. The ByteBuffer should
   *     not be modified after the construction of an {@link InterpreterApi} instance. The {@code
   *     ByteBuffer} can be either a {@code MappedByteBuffer} that memory-maps a model file, or a
   *     direct {@code ByteBuffer} of nativeOrder() that contains the bytes content of a model.
   * @param options A set of options for customizing interpreter behavior.
   * @throws IllegalArgumentException if {@code byteBuffer} is not a {@code MappedByteBuffer} nor a
   *     direct {@code ByteBuffer} of nativeOrder.
   */
  @SuppressWarnings("StaticOrDefaultInterfaceMethod")
  static InterpreterApi create(@NonNull ByteBuffer byteBuffer, InterpreterApi.Options options) {
    TfLiteRuntime runtime = (options == null ? null : options.getRuntime());
    InterpreterFactoryApi factory = TensorFlowLite.getFactory(runtime);
    return factory.create(byteBuffer, options);
  }

  /**
   * Runs model inference if the model takes only one input, and provides only one output.
   *
   * <p>Warning: The API is more efficient if a {@code Buffer} (preferably direct, but not required)
   * is used as the input/output data type. Please consider using {@code Buffer} to feed and fetch
   * primitive data for better performance. The following concrete {@code Buffer} types are
   * supported:
   *
   * <ul>
   *   <li>{@code ByteBuffer} - compatible with any underlying primitive Tensor type.
   *   <li>{@code FloatBuffer} - compatible with float Tensors.
   *   <li>{@code IntBuffer} - compatible with int32 Tensors.
   *   <li>{@code LongBuffer} - compatible with int64 Tensors.
   * </ul>
   *
   * Note that boolean types are only supported as arrays, not {@code Buffer}s, or as scalar inputs.
   *
   * @param input an array or multidimensional array, or a {@code Buffer} of primitive types
   *     including int, float, long, and byte. {@code Buffer} is the preferred way to pass large
   *     input data for primitive types, whereas string types require using the (multi-dimensional)
   *     array input path. When a {@code Buffer} is used, its content should remain unchanged until
   *     model inference is done, and the caller must ensure that the {@code Buffer} is at the
   *     appropriate read position. A {@code null} value is allowed only if the caller is using a
   *     {@link Delegate} that allows buffer handle interop, and such a buffer has been bound to the
   *     input {@link Tensor}.
   * @param output a multidimensional array of output data, or a {@code Buffer} of primitive types
   *     including int, float, long, and byte. When a {@code Buffer} is used, the caller must ensure
   *     that it is set the appropriate write position. A null value is allowed, and is useful for
   *     certain cases, e.g., if the caller is using a {@link Delegate} that allows buffer handle
   *     interop, and such a buffer has been bound to the output {@link Tensor} (see also <a
   *     href="https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Interpreter.Options#setAllowBufferHandleOutput(boolean)">Interpreter.Options#setAllowBufferHandleOutput(boolean)</a>),
   *     or if the graph has dynamically shaped outputs and the caller must query the output {@link
   *     Tensor} shape after inference has been invoked, fetching the data directly from the output
   *     tensor (via {@link Tensor#asReadOnlyBuffer()}).
   * @throws IllegalArgumentException if {@code input} is null or empty, or if an error occurs when
   *     running inference.
   * @throws IllegalArgumentException (EXPERIMENTAL, subject to change) if the inference is
   *     interrupted by {@code setCancelled(true)}.
   */
  void run(Object input, Object output);

  /**
   * Runs model inference if the model takes multiple inputs, or returns multiple outputs.
   *
   * <p>Warning: The API is more efficient if {@code Buffer}s (preferably direct, but not required)
   * are used as the input/output data types. Please consider using {@code Buffer} to feed and fetch
   * primitive data for better performance. The following concrete {@code Buffer} types are
   * supported:
   *
   * <ul>
   *   <li>{@code ByteBuffer} - compatible with any underlying primitive Tensor type.
   *   <li>{@code FloatBuffer} - compatible with float Tensors.
   *   <li>{@code IntBuffer} - compatible with int32 Tensors.
   *   <li>{@code LongBuffer} - compatible with int64 Tensors.
   * </ul>
   *
   * Note that boolean types are only supported as arrays, not {@code Buffer}s, or as scalar inputs.
   *
   * <p>Note: {@code null} values for invididual elements of {@code inputs} and {@code outputs} is
   * allowed only if the caller is using a {@link Delegate} that allows buffer handle interop, and
   * such a buffer has been bound to the corresponding input or output {@link Tensor}(s).
   *
   * @param inputs an array of input data. The inputs should be in the same order as inputs of the
   *     model. Each input can be an array or multidimensional array, or a {@code Buffer} of
   *     primitive types including int, float, long, and byte. {@code Buffer} is the preferred way
   *     to pass large input data, whereas string types require using the (multi-dimensional) array
   *     input path. When {@code Buffer} is used, its content should remain unchanged until model
   *     inference is done, and the caller must ensure that the {@code Buffer} is at the appropriate
   *     read position.
   * @param outputs a map mapping output indices to multidimensional arrays of output data or {@code
   *     Buffer}s of primitive types including int, float, long, and byte. It only needs to keep
   *     entries for the outputs to be used. When a {@code Buffer} is used, the caller must ensure
   *     that it is set the appropriate write position. The map may be empty for cases where either
   *     buffer handles are used for output tensor data, or cases where the outputs are dynamically
   *     shaped and the caller must query the output {@link Tensor} shape after inference has been
   *     invoked, fetching the data directly from the output tensor (via {@link
   *     Tensor#asReadOnlyBuffer()}).
   * @throws IllegalArgumentException if {@code inputs} is null or empty, if {@code outputs} is
   *     null, or if an error occurs when running inference.
   */
  void runForMultipleInputsOutputs(
      Object @NonNull [] inputs, @NonNull Map<Integer, Object> outputs);

  /**
   * Explicitly updates allocations for all tensors, if necessary.
   *
   * <p>This will propagate shapes and memory allocations for dependent tensors using the input
   * tensor shape(s) as given.
   *
   * <p>Note: This call is *purely optional*. Tensor allocation will occur automatically during
   * execution if any input tensors have been resized. This call is most useful in determining the
   * shapes for any output tensors before executing the graph, e.g.,
   *
   * <pre> {@code
   * interpreter.resizeInput(0, new int[]{1, 4, 4, 3}));
   * interpreter.allocateTensors();
   * FloatBuffer input = FloatBuffer.allocate(interpreter.getInputTensor(0).numElements());
   * // Populate inputs...
   * FloatBuffer output = FloatBuffer.allocate(interpreter.getOutputTensor(0).numElements());
   * interpreter.run(input, output)
   * // Process outputs...}</pre>
   *
   * <p>Note: Some graphs have dynamically shaped outputs, in which case the output shape may not
   * fully propagate until inference is executed.
   *
   * @throws IllegalStateException if the graph's tensors could not be successfully allocated.
   */
  void allocateTensors();

  /**
   * Resizes idx-th input of the native model to the given dims.
   *
   * @throws IllegalArgumentException if {@code idx} is negative or is not smaller than the number
   *     of model inputs; or if error occurs when resizing the idx-th input.
   */
  void resizeInput(int idx, @NonNull int[] dims);

  /**
   * Resizes idx-th input of the native model to the given dims.
   *
   * <p>When `strict` is True, only unknown dimensions can be resized. Unknown dimensions are
   * indicated as `-1` in the array returned by `Tensor.shapeSignature()`.
   *
   * @throws IllegalArgumentException if {@code idx} is negative or is not smaller than the number
   *     of model inputs; or if error occurs when resizing the idx-th input. Additionally, the error
   *     occurs when attempting to resize a tensor with fixed dimensions when `strict` is True.
   */
  void resizeInput(int idx, @NonNull int[] dims, boolean strict);

  /** Gets the number of input tensors. */
  int getInputTensorCount();

  /**
   * Gets index of an input given the op name of the input.
   *
   * @throws IllegalArgumentException if {@code opName} does not match any input in the model used
   *     to initialize the interpreter.
   */
  int getInputIndex(String opName);

  /**
   * Gets the Tensor associated with the provided input index.
   *
   * @throws IllegalArgumentException if {@code inputIndex} is negative or is not smaller than the
   *     number of model inputs.
   */
  Tensor getInputTensor(int inputIndex);

  /** Gets the number of output Tensors. */
  int getOutputTensorCount();

  /**
   * Gets index of an output given the op name of the output.
   *
   * @throws IllegalArgumentException if {@code opName} does not match any output in the model used
   *     to initialize the interpreter.
   */
  int getOutputIndex(String opName);

  /**
   * Gets the Tensor associated with the provided output index.
   *
   * <p>Note: Output tensor details (e.g., shape) may not be fully populated until after inference
   * is executed. If you need updated details *before* running inference (e.g., after resizing an
   * input tensor, which may invalidate output tensor shapes), use {@link #allocateTensors()} to
   * explicitly trigger allocation and shape propagation. Note that, for graphs with output shapes
   * that are dependent on input *values*, the output shape may not be fully determined until
   * running inference.
   *
   * @throws IllegalArgumentException if {@code outputIndex} is negative or is not smaller than the
   *     number of model outputs.
   */
  Tensor getOutputTensor(int outputIndex);

  /**
   * Returns native inference timing.
   *
   * @throws IllegalArgumentException if the model is not initialized by the interpreter.
   */
  Long getLastNativeInferenceDurationNanoseconds();

  /** Release resources associated with the {@code InterpreterApi} instance. */
  @Override
  void close();
}
