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
import java.util.Arrays;
import java.util.Map;
import org.checkerframework.checker.nullness.qual.NonNull;

/**
 * Driver class to drive model inference with TensorFlow Lite.
 *
 * <p>Note: If you don't need access to any of the "experimental" API features below, prefer to use
 * InterpreterApi and InterpreterFactory rather than using Interpreter directly.
 *
 * <p>A {@code Interpreter} encapsulates a pre-trained TensorFlow Lite model, in which operations
 * are executed for model inference.
 *
 * <p>For example, if a model takes only one input and returns only one output:
 *
 * <pre>{@code
 * try (Interpreter interpreter = new Interpreter(file_of_a_tensorflowlite_model)) {
 *   interpreter.run(input, output);
 * }
 * }</pre>
 *
 * <p>If a model takes multiple inputs or outputs:
 *
 * <pre>{@code
 * Object[] inputs = {input0, input1, ...};
 * Map<Integer, Object> map_of_indices_to_outputs = new HashMap<>();
 * FloatBuffer ith_output = FloatBuffer.allocateDirect(3 * 2 * 4);  // Float tensor, shape 3x2x4.
 * ith_output.order(ByteOrder.nativeOrder());
 * map_of_indices_to_outputs.put(i, ith_output);
 * try (Interpreter interpreter = new Interpreter(file_of_a_tensorflowlite_model)) {
 *   interpreter.runForMultipleInputsOutputs(inputs, map_of_indices_to_outputs);
 * }
 * }</pre>
 *
 * <p>If a model takes or produces string tensors:
 *
 * <pre>{@code
 * String[] input = {"foo", "bar"};  // Input tensor shape is [2].
 * String[] output = new String[3][2];  // Output tensor shape is [3, 2].
 * try (Interpreter interpreter = new Interpreter(file_of_a_tensorflowlite_model)) {
 *   interpreter.runForMultipleInputsOutputs(input, output);
 * }
 * }</pre>
 *
 * <p>Orders of inputs and outputs are determined when converting TensorFlow model to TensorFlowLite
 * model with Toco, as are the default shapes of the inputs.
 *
 * <p>When inputs are provided as (multi-dimensional) arrays, the corresponding input tensor(s) will
 * be implicitly resized according to that array's shape. When inputs are provided as {@code Buffer}
 * types, no implicit resizing is done; the caller must ensure that the {@code Buffer} byte size
 * either matches that of the corresponding tensor, or that they first resize the tensor via {@link
 * #resizeInput(int, int[])}. Tensor shape and type information can be obtained via the {@link
 * Tensor} class, available via {@link #getInputTensor(int)} and {@link #getOutputTensor(int)}.
 *
 * <p><b>WARNING:</b>{@code Interpreter} instances are <b>not</b> thread-safe. A {@code Interpreter}
 * owns resources that <b>must</b> be explicitly freed by invoking {@link #close()}
 *
 * <p>The TFLite library is built against NDK API 19. It may work for Android API levels below 19,
 * but is not guaranteed.
 */
public final class Interpreter extends InterpreterImpl implements InterpreterApi {

  /** An options class for controlling runtime interpreter behavior. */
  public static class Options extends InterpreterImpl.Options {
    public Options() {}

    public Options(InterpreterApi.Options options) {
      super(options);
    }

    Options(InterpreterImpl.Options options) {
      super(options);
    }

    @Override
    public Options setNumThreads(int numThreads) {
      super.setNumThreads(numThreads);
      return this;
    }

    @Override
    public Options setUseNNAPI(boolean useNNAPI) {
      super.setUseNNAPI(useNNAPI);
      return this;
    }

    /**
     * Sets whether to allow float16 precision for FP32 calculation when possible. Defaults to false
     * (disallow).
     *
     * @deprecated Prefer using <a
     *     href="https://github.com/tensorflow/tensorflow/blob/5dc7f6981fdaf74c8c5be41f393df705841fb7c5/tensorflow/lite/delegates/nnapi/java/src/main/java/org/tensorflow/lite/nnapi/NnApiDelegate.java#L127">NnApiDelegate.Options#setAllowFp16(boolean
     *     enable)</a>.
     */
    @Deprecated
    public Options setAllowFp16PrecisionForFp32(boolean allow) {
      this.allowFp16PrecisionForFp32 = allow;
      return this;
    }

    @Override
    public Options addDelegate(Delegate delegate) {
      super.addDelegate(delegate);
      return this;
    }

    @Override
    public Options addDelegateFactory(DelegateFactory delegateFactory) {
      super.addDelegateFactory(delegateFactory);
      return this;
    }

    /**
     * Advanced: Set if buffer handle output is allowed.
     *
     * <p>When a {@link Delegate} supports hardware acceleration, the interpreter will make the data
     * of output tensors available in the CPU-allocated tensor buffers by default. If the client can
     * consume the buffer handle directly (e.g. reading output from OpenGL texture), it can set this
     * flag to false, avoiding the copy of data to the CPU buffer. The delegate documentation should
     * indicate whether this is supported and how it can be used.
     *
     * <p>WARNING: This is an experimental interface that is subject to change.
     */
    public Options setAllowBufferHandleOutput(boolean allow) {
      this.allowBufferHandleOutput = allow;
      return this;
    }

    @Override
    public Options setCancellable(boolean allow) {
      super.setCancellable(allow);
      return this;
    }

    /**
     * Experimental: Disable an optimized set of CPU kernels (provided by XNNPACK).
     *
     * <p>Disabling this flag will disable use of a highly optimized set of CPU kernels provided via
     * the XNNPACK delegate. Currently, this is restricted to a subset of floating point operations.
     * See
     * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/xnnpack/README.md
     * for more details.
     *
     * <p>WARNING: This is an experimental interface that is subject to change.
     */
    public Options setUseXNNPACK(boolean useXNNPACK) {
      this.useXNNPACK = useXNNPACK;
      return this;
    }

    @Override
    public Options setRuntime(InterpreterApi.Options.TfLiteRuntime runtime) {
      super.setRuntime(runtime);
      return this;
    }
  }

  /**
   * Initializes an {@code Interpreter}.
   *
   * @param modelFile a File of a pre-trained TF Lite model.
   * @throws IllegalArgumentException if {@code modelFile} does not encode a valid TensorFlow Lite
   *     model.
   */
  public Interpreter(@NonNull File modelFile) {
    this(modelFile, /*options = */ null);
  }

  /**
   * Initializes an {@code Interpreter} and specifies options for customizing interpreter behavior.
   *
   * @param modelFile a file of a pre-trained TF Lite model
   * @param options a set of options for customizing interpreter behavior
   * @throws IllegalArgumentException if {@code modelFile} does not encode a valid TensorFlow Lite
   *     model.
   */
  public Interpreter(@NonNull File modelFile, Options options) {
    this(new NativeInterpreterWrapperExperimental(modelFile.getAbsolutePath(), options));
  }

  /**
   * Initializes an {@code Interpreter} with a {@code ByteBuffer} of a model file.
   *
   * <p>The ByteBuffer should not be modified after the construction of a {@code Interpreter}. The
   * {@code ByteBuffer} can be either a {@code MappedByteBuffer} that memory-maps a model file, or a
   * direct {@code ByteBuffer} of nativeOrder() that contains the bytes content of a model.
   *
   * @throws IllegalArgumentException if {@code byteBuffer} is not a {@code MappedByteBuffer} nor a
   *     direct {@code ByteBuffer} of nativeOrder.
   */
  public Interpreter(@NonNull ByteBuffer byteBuffer) {
    this(byteBuffer, /* options= */ null);
  }

  /**
   * Initializes an {@code Interpreter} with a {@code ByteBuffer} of a model file and a set of
   * custom {@link Interpreter.Options}.
   *
   * <p>The {@code ByteBuffer} should not be modified after the construction of an {@code
   * Interpreter}. The {@code ByteBuffer} can be either a {@code MappedByteBuffer} that memory-maps
   * a model file, or a direct {@code ByteBuffer} of nativeOrder() that contains the bytes content
   * of a model.
   *
   * @throws IllegalArgumentException if {@code byteBuffer} is not a {@code MappedByteBuffer} nor a
   *     direct {@code ByteBuffer} of nativeOrder.
   */
  public Interpreter(@NonNull ByteBuffer byteBuffer, Options options) {
    this(new NativeInterpreterWrapperExperimental(byteBuffer, options));
  }

  private Interpreter(NativeInterpreterWrapperExperimental wrapper) {
    super(wrapper);
    wrapperExperimental = wrapper;
    signatureKeyList = getSignatureKeys();
  }

  /**
   * Runs model inference based on SignatureDef provided through {@code signatureKey}.
   *
   * <p>See {@link Interpreter#run(Object, Object)} for more details on the allowed input and output
   * data types.
   *
   * <p>WARNING: This is an experimental API and subject to change.
   *
   * @param inputs A map from input name in the SignatureDef to an input object.
   * @param outputs A map from output name in SignatureDef to output data. This may be empty if the
   *     caller wishes to query the {@link Tensor} data directly after inference (e.g., if the
   *     output shape is dynamic, or output buffer handles are used).
   * @param signatureKey Signature key identifying the SignatureDef.
   * @throws IllegalArgumentException if {@code inputs} is null or empty, if {@code outputs} or
   *     {@code signatureKey} is null, or if an error occurs when running inference.
   */
  public void runSignature(
      @NonNull Map<String, Object> inputs,
      @NonNull Map<String, Object> outputs,
      String signatureKey) {
    checkNotClosed();
    if (signatureKey == null && signatureKeyList.length == 1) {
      signatureKey = signatureKeyList[0];
    }
    if (signatureKey == null) {
      throw new IllegalArgumentException(
          "Input error: SignatureDef signatureKey should not be null. null is only allowed if the"
              + " model has a single Signature. Available Signatures: "
              + Arrays.toString(signatureKeyList));
    }
    wrapper.runSignature(inputs, outputs, signatureKey);
  }

  /**
   * Same as {@link #runSignature(Map, Map, String)} but doesn't require passing a signatureKey,
   * assuming the model has one SignatureDef. If the model has more than one SignatureDef it will
   * throw an exception.
   *
   * <p>WARNING: This is an experimental API and subject to change.
   */
  public void runSignature(
      @NonNull Map<String, Object> inputs, @NonNull Map<String, Object> outputs) {
    checkNotClosed();
    runSignature(inputs, outputs, null);
  }

  /**
   * Gets the Tensor associated with the provided input name and signature method name.
   *
   * <p>WARNING: This is an experimental API and subject to change.
   *
   * @param inputName Input name in the signature.
   * @param signatureKey Signature key identifying the SignatureDef, can be null if the model has
   *     one signature.
   * @throws IllegalArgumentException if {@code inputName} or {@code signatureKey} is null or empty,
   *     or invalid name provided.
   */
  public Tensor getInputTensorFromSignature(String inputName, String signatureKey) {
    checkNotClosed();
    if (signatureKey == null && signatureKeyList.length == 1) {
      signatureKey = signatureKeyList[0];
    }
    if (signatureKey == null) {
      throw new IllegalArgumentException(
          "Input error: SignatureDef signatureKey should not be null. null is only allowed if the"
              + " model has a single Signature. Available Signatures: "
              + Arrays.toString(signatureKeyList));
    }
    return wrapper.getInputTensor(inputName, signatureKey);
  }

  /**
   * Gets the list of SignatureDef exported method names available in the model.
   *
   * <p>WARNING: This is an experimental API and subject to change.
   */
  public String[] getSignatureKeys() {
    checkNotClosed();
    return wrapper.getSignatureKeys();
  }

  /**
   * Gets the list of SignatureDefs inputs for method {@code signatureKey}.
   *
   * <p>WARNING: This is an experimental API and subject to change.
   */
  public String[] getSignatureInputs(String signatureKey) {
    checkNotClosed();
    return wrapper.getSignatureInputs(signatureKey);
  }

  /**
   * Gets the list of SignatureDefs outputs for method {@code signatureKey}.
   *
   * <p>WARNING: This is an experimental API and subject to change.
   */
  public String[] getSignatureOutputs(String signatureKey) {
    checkNotClosed();
    return wrapper.getSignatureOutputs(signatureKey);
  }

  /**
   * Gets the Tensor associated with the provided output name in specific signature method.
   *
   * <p>Note: Output tensor details (e.g., shape) may not be fully populated until after inference
   * is executed. If you need updated details *before* running inference (e.g., after resizing an
   * input tensor, which may invalidate output tensor shapes), use {@link #allocateTensors()} to
   * explicitly trigger allocation and shape propagation. Note that, for graphs with output shapes
   * that are dependent on input *values*, the output shape may not be fully determined until
   * running inference.
   *
   * <p>WARNING: This is an experimental API and subject to change.
   *
   * @param outputName Output name in the signature.
   * @param signatureKey Signature key identifying the SignatureDef, can be null if the model has
   *     one signature.
   * @throws IllegalArgumentException if {@code outputName} or {@code signatureKey} is null or
   *     empty, or invalid name provided.
   */
  public Tensor getOutputTensorFromSignature(String outputName, String signatureKey) {
    checkNotClosed();
    if (signatureKey == null && signatureKeyList.length == 1) {
      signatureKey = signatureKeyList[0];
    }
    if (signatureKey == null) {
      throw new IllegalArgumentException(
          "Input error: SignatureDef signatureKey should not be null. null is only allowed if the"
              + " model has a single Signature. Available Signatures: "
              + Arrays.toString(signatureKeyList));
    }
    return wrapper.getOutputTensor(outputName, signatureKey);
  }

  /**
   * Advanced: Resets all variable tensors to the default value.
   *
   * <p>If a variable tensor doesn't have an associated buffer, it will be reset to zero.
   *
   * <p>WARNING: This is an experimental API and subject to change.
   */
  public void resetVariableTensors() {
    checkNotClosed();
    wrapperExperimental.resetVariableTensors();
  }

  /**
   * Advanced: Interrupts inference in the middle of a call to {@link Interpreter#run}.
   *
   * <p>A cancellation flag will be set to true when this function gets called. The interpreter will
   * check the flag between Op invocations, and if it's {@code true}, the interpreter will stop
   * execution. The interpreter will remain a cancelled state until explicitly "uncancelled" by
   * {@code setCancelled(false)}.
   *
   * <p>WARNING: This is an experimental API and subject to change.
   *
   * @param cancelled {@code true} to cancel inference in a best-effort way; {@code false} to
   *     resume.
   * @throws IllegalStateException if the interpreter is not initialized with the cancellable
   *     option, which is by default off.
   * @see Interpreter.Options#setCancellable(boolean).
   */
  public void setCancelled(boolean cancelled) {
    wrapper.setCancelled(cancelled);
  }

  private final NativeInterpreterWrapperExperimental wrapperExperimental;
  private final String[] signatureKeyList;
}
