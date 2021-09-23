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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.tensorflow.lite.nnapi.NnApiDelegate;

/**
 * An internal wrapper that wraps native interpreter and controls model execution.
 *
 * <p><b>WARNING:</b> Resources consumed by the {@code NativeInterpreterWrapper} object must be
 * explicitly freed by invoking the {@link #close()} method when the {@code
 * NativeInterpreterWrapper} object is no longer needed.
 *
 * <p>Note: This class is not thread safe.
 */
class NativeInterpreterWrapper implements AutoCloseable {

  NativeInterpreterWrapper(String modelPath) {
    this(modelPath, /* options= */ null);
  }

  NativeInterpreterWrapper(ByteBuffer byteBuffer) {
    this(byteBuffer, /* options= */ null);
  }

  NativeInterpreterWrapper(String modelPath, InterpreterImpl.Options options) {
    TensorFlowLite.init();
    long errorHandle = createErrorReporter(ERROR_BUFFER_SIZE);
    long modelHandle = createModel(modelPath, errorHandle);
    init(errorHandle, modelHandle, options);
  }

  NativeInterpreterWrapper(ByteBuffer buffer, InterpreterImpl.Options options) {
    TensorFlowLite.init();
    if (buffer == null
        || (!(buffer instanceof MappedByteBuffer)
            && (!buffer.isDirect() || buffer.order() != ByteOrder.nativeOrder()))) {
      throw new IllegalArgumentException(
          "Model ByteBuffer should be either a MappedByteBuffer of the model file, or a direct "
              + "ByteBuffer using ByteOrder.nativeOrder() which contains bytes of model content.");
    }
    this.modelByteBuffer = buffer;
    long errorHandle = createErrorReporter(ERROR_BUFFER_SIZE);
    long modelHandle = createModelWithBuffer(modelByteBuffer, errorHandle);
    init(errorHandle, modelHandle, options);
  }

  private void init(long errorHandle, long modelHandle, InterpreterImpl.Options options) {
    if (options == null) {
      options = new InterpreterImpl.Options();
    }
    this.errorHandle = errorHandle;
    this.modelHandle = modelHandle;
    // First create the interpreter without delegates.
    // We need this in order to figure out whether the model contains any unresolved flex ops.
    // (Alternatively, we could determine this without needing to recreate the interpreter
    // by passing the tflite::Model in to here, and then traversing that?)
    this.interpreterHandle = createInterpreter(modelHandle, errorHandle, options.numThreads);
    this.originalGraphHasUnresolvedFlexOp = hasUnresolvedFlexOp(interpreterHandle);
    addDelegates(options);
    // TODO(b/187920750): uncomment this when createInterpreter is modified to use
    // InterpreterBuilder::AddDelegate.
    // if (!delegates.isEmpty()) {
    //   // If there are any delegates enabled, recreate the interpreter with those delegates.
    //   delete(/* errorHandle= */ 0, /* modelHandle= */ 0, this.interpreterHandle);
    //   this.interpreterHandle = createInterpreter(modelHandle, errorHandle, options.numThreads);
    // }
    if (options.allowFp16PrecisionForFp32 != null) {
      allowFp16PrecisionForFp32(
          interpreterHandle, options.allowFp16PrecisionForFp32.booleanValue());
    }
    applyDelegates(options);
    if (options.allowBufferHandleOutput != null) {
      allowBufferHandleOutput(interpreterHandle, options.allowBufferHandleOutput.booleanValue());
    }
    if (options.allowCancellation != null && options.allowCancellation) {
      this.cancellationFlagHandle = createCancellationFlag(interpreterHandle);
    }
    this.inputTensors = new TensorImpl[getInputCount(interpreterHandle)];
    this.outputTensors = new TensorImpl[getOutputCount(interpreterHandle)];
    if (options.allowFp16PrecisionForFp32 != null) {
      allowFp16PrecisionForFp32(
          interpreterHandle, options.allowFp16PrecisionForFp32.booleanValue());
    }
    if (options.allowBufferHandleOutput != null) {
      allowBufferHandleOutput(interpreterHandle, options.allowBufferHandleOutput.booleanValue());
    }
    allocateTensors(interpreterHandle, errorHandle);
    this.isMemoryAllocated = true;
  }

  /** Releases resources associated with this {@code NativeInterpreterWrapper}. */
  @Override
  public void close() {
    // Close the tensors first as they may reference the native interpreter.
    for (int i = 0; i < inputTensors.length; ++i) {
      if (inputTensors[i] != null) {
        inputTensors[i].close();
        inputTensors[i] = null;
      }
    }
    for (int i = 0; i < outputTensors.length; ++i) {
      if (outputTensors[i] != null) {
        outputTensors[i].close();
        outputTensors[i] = null;
      }
    }
    delete(errorHandle, modelHandle, interpreterHandle);
    deleteCancellationFlag(cancellationFlagHandle);
    errorHandle = 0;
    modelHandle = 0;
    interpreterHandle = 0;
    cancellationFlagHandle = 0;
    modelByteBuffer = null;
    inputsIndexes = null;
    outputsIndexes = null;
    isMemoryAllocated = false;
    delegates.clear();
    for (AutoCloseable ownedDelegate : ownedDelegates) {
      try {
        ownedDelegate.close();
      } catch (Exception e) {
        System.err.println("Failed to close flex delegate: " + e);
      }
    }
    ownedDelegates.clear();
  }

  /** Runs model inference based on SignatureDef provided through {@code signatureKey}. */
  public void runSignature(
      Map<String, Object> inputs, Map<String, Object> outputs, String signatureKey) {
    inferenceDurationNanoseconds = -1;
    if (inputs == null || inputs.isEmpty()) {
      throw new IllegalArgumentException("Input error: Inputs should not be null or empty.");
    }
    if (outputs == null) {
      throw new IllegalArgumentException("Input error: Outputs should not be null.");
    }
    NativeSignatureRunnerWrapper signatureRunnerWrapper = getSignatureRunnerWrapper(signatureKey);
    int subgraphIndex = signatureRunnerWrapper.getSubgraphIndex();
    if (subgraphIndex == 0) {
      // Run the primary subgraph and update the cached tensor objects.
      initTensorIndexesMaps();
      // Map inputs/output to input indexes.
      Object[] inputsList = new Object[inputs.size()];
      for (Map.Entry<String, Object> input : inputs.entrySet()) {
        inputsList[signatureRunnerWrapper.getInputIndex(input.getKey())] = input.getValue();
      }
      Map<Integer, Object> outputsWithOutputIndex = new TreeMap<>();
      for (Map.Entry<String, Object> output : outputs.entrySet()) {
        outputsWithOutputIndex.put(
            signatureRunnerWrapper.getOutputIndex(output.getKey()), output.getValue());
      }
      run(inputsList, outputsWithOutputIndex);
      return;
    }

    for (Map.Entry<String, Object> input : inputs.entrySet()) {
      TensorImpl tensor = getInputTensor(input.getKey(), signatureKey);
      int[] newShape = tensor.getInputShapeIfDifferent(input.getValue());
      if (newShape != null) {
        signatureRunnerWrapper.resizeInput(input.getKey(), newShape);
      }
    }

    signatureRunnerWrapper.allocateTensorsIfNeeded();

    for (Map.Entry<String, Object> input : inputs.entrySet()) {
      signatureRunnerWrapper.getInputTensor(input.getKey()).setTo(input.getValue());
    }

    long inferenceStartNanos = System.nanoTime();
    signatureRunnerWrapper.invoke();
    long inferenceDurationNanoseconds = System.nanoTime() - inferenceStartNanos;

    for (Map.Entry<String, Object> output : outputs.entrySet()) {
      // Null output placeholders are allowed and ignored.
      if (output.getValue() != null) {
        signatureRunnerWrapper.getOutputTensor(output.getKey()).copyTo(output.getValue());
      }
    }

    // Only set if the entire operation succeeds.
    this.inferenceDurationNanoseconds = inferenceDurationNanoseconds;
  }

  /** Sets inputs, runs model inference and returns outputs. */
  void run(Object[] inputs, Map<Integer, Object> outputs) {
    inferenceDurationNanoseconds = -1;
    if (inputs == null || inputs.length == 0) {
      throw new IllegalArgumentException("Input error: Inputs should not be null or empty.");
    }
    if (outputs == null) {
      throw new IllegalArgumentException("Input error: Outputs should not be null.");
    }

    // TODO(b/80431971): Remove implicit resize after deprecating multi-dimensional array inputs.
    // Rather than forcing an immediate resize + allocation if an input's shape differs, we first
    // flush all resizes, avoiding redundant allocations.
    for (int i = 0; i < inputs.length; ++i) {
      TensorImpl tensor = getInputTensor(i);
      int[] newShape = tensor.getInputShapeIfDifferent(inputs[i]);
      if (newShape != null) {
        resizeInput(i, newShape);
      }
    }

    boolean allocatedTensors = allocateTensorsIfNeeded();

    for (int i = 0; i < inputs.length; ++i) {
      getInputTensor(i).setTo(inputs[i]);
    }

    long inferenceStartNanos = System.nanoTime();
    run(interpreterHandle, errorHandle);
    long inferenceDurationNanoseconds = System.nanoTime() - inferenceStartNanos;

    // Allocation can trigger dynamic resizing of output tensors, so refresh all output shapes.
    if (allocatedTensors) {
      for (int i = 0; i < outputTensors.length; ++i) {
        if (outputTensors[i] != null) {
          outputTensors[i].refreshShape();
        }
      }
    }
    for (Map.Entry<Integer, Object> output : outputs.entrySet()) {
      // Null output placeholders are allowed and ignored.
      if (output.getValue() != null) {
        getOutputTensor(output.getKey()).copyTo(output.getValue());
      }
    }

    // Only set if the entire operation succeeds.
    this.inferenceDurationNanoseconds = inferenceDurationNanoseconds;
  }

  /** Resizes dimensions of a specific input. */
  void resizeInput(int idx, int[] dims) {
    resizeInput(idx, dims, false);
  }

  /** Resizes dimensions of a specific input. */
  void resizeInput(int idx, int[] dims, boolean strict) {
    if (resizeInput(interpreterHandle, errorHandle, idx, dims, strict)) {
      // Tensor allocation is deferred until either an explicit `allocateTensors()` call or
      // `invoke()` avoiding redundant allocations if multiple tensors are simultaneosly resized.
      isMemoryAllocated = false;
      if (inputTensors[idx] != null) {
        inputTensors[idx].refreshShape();
      }
    }
  }

  /** Triggers explicit allocation of tensors. */
  void allocateTensors() {
    allocateTensorsIfNeeded();
  }

  /**
   * Allocates tensor memory space in the given subgraph and returns true when allocation happens
   */
  private boolean allocateTensorsIfNeeded() {
    if (isMemoryAllocated) {
      return false;
    }

    isMemoryAllocated = true;
    allocateTensors(interpreterHandle, errorHandle);
    for (int i = 0; i < outputTensors.length; ++i) {
      if (outputTensors[i] != null) {
        outputTensors[i].refreshShape();
      }
    }
    return true;
  }

  /** Gets index of an input given its name. */
  int getInputIndex(String name) {
    if (inputsIndexes == null) {
      String[] names = getInputNames(interpreterHandle);
      inputsIndexes = new HashMap<>();
      if (names != null) {
        for (int i = 0; i < names.length; ++i) {
          inputsIndexes.put(names[i], i);
        }
      }
    }
    if (inputsIndexes.containsKey(name)) {
      return inputsIndexes.get(name);
    } else {
      throw new IllegalArgumentException(
          String.format(
              "Input error: '%s' is not a valid name for any input. Names of inputs and their "
                  + "indexes are %s",
              name, inputsIndexes));
    }
  }

  /** Initializes mapping from tensor index to input/output index. * */
  private void initTensorIndexesMaps() {
    if (tensorToInputsIndexes != null) {
      return;
    }
    tensorToInputsIndexes = new HashMap<>();
    tensorToOutputsIndexes = new HashMap<>();
    int inputCount = getInputTensorCount();
    for (int i = 0; i < inputCount; ++i) {
      int tensorIndex = getInputTensorIndex(interpreterHandle, i);
      tensorToInputsIndexes.put(tensorIndex, i);
    }
    int outputCount = getOutputTensorCount();
    for (int i = 0; i < outputCount; ++i) {
      int tensorIndex = getOutputTensorIndex(interpreterHandle, i);
      tensorToOutputsIndexes.put(tensorIndex, i);
    }
  }

  /** Gets index of an output given its name. */
  int getOutputIndex(String name) {
    if (outputsIndexes == null) {
      String[] names = getOutputNames(interpreterHandle);
      outputsIndexes = new HashMap<>();
      if (names != null) {
        for (int i = 0; i < names.length; ++i) {
          outputsIndexes.put(names[i], i);
        }
      }
    }
    if (outputsIndexes.containsKey(name)) {
      return outputsIndexes.get(name);
    } else {
      throw new IllegalArgumentException(
          String.format(
              "Input error: '%s' is not a valid name for any output. Names of outputs and their "
                  + "indexes are %s",
              name, outputsIndexes));
    }
  }

  /**
   * Gets the last inference duration in nanoseconds. It returns null if there is no previous
   * inference run or the last inference run failed.
   */
  Long getLastNativeInferenceDurationNanoseconds() {
    return (inferenceDurationNanoseconds < 0) ? null : inferenceDurationNanoseconds;
  }

  /** Gets the number of input tensors. */
  int getInputTensorCount() {
    return inputTensors.length;
  }

  /**
   * Gets the input {@link TensorImpl} for the provided input index.
   *
   * @throws IllegalArgumentException if the input index is invalid.
   */
  TensorImpl getInputTensor(int index) {
    if (index < 0 || index >= inputTensors.length) {
      throw new IllegalArgumentException("Invalid input Tensor index: " + index);
    }
    TensorImpl inputTensor = inputTensors[index];
    if (inputTensor == null) {
      inputTensor =
          inputTensors[index] =
              TensorImpl.fromIndex(
                  interpreterHandle, getInputTensorIndex(interpreterHandle, index));
    }
    return inputTensor;
  }

  /**
   * Gets the input {@link TensorImpl} given the tensor name and method in the signature.
   *
   * @throws IllegalArgumentException if the input name is invalid.
   */
  TensorImpl getInputTensor(String inputName, String signatureKey) {
    if (inputName == null) {
      throw new IllegalArgumentException("Invalid input tensor name provided (null)");
    }
    NativeSignatureRunnerWrapper signatureRunnerWrapper = getSignatureRunnerWrapper(signatureKey);
    int subgraphIndex = signatureRunnerWrapper.getSubgraphIndex();
    if (subgraphIndex > 0) {
      return signatureRunnerWrapper.getInputTensor(inputName);
    }

    int inputIndex = signatureRunnerWrapper.getInputIndex(inputName);
    return getInputTensor(inputIndex);
  }

  /** Gets the keys of SignatureDefs available in the model, if any. */
  public String[] getSignatureKeys() {
    return getSignatureKeys(interpreterHandle);
  }

  /** Gets the list of SignatureDefs inputs for method {@code signatureKey} */
  String[] getSignatureInputs(String signatureKey) {
    return getSignatureRunnerWrapper(signatureKey).inputNames();
  }

  /** Gets the list of SignatureDefs outputs for method {@code signatureKey} */
  String[] getSignatureOutputs(String signatureKey) {
    return getSignatureRunnerWrapper(signatureKey).outputNames();
  }

  /** Gets the number of output tensors. */
  int getOutputTensorCount() {
    return outputTensors.length;
  }

  /**
   * Gets the output {@link TensorImpl} for the provided output index.
   *
   * @throws IllegalArgumentException if the output index is invalid.
   */
  TensorImpl getOutputTensor(int index) {
    if (index < 0 || index >= outputTensors.length) {
      throw new IllegalArgumentException("Invalid output Tensor index: " + index);
    }
    TensorImpl outputTensor = outputTensors[index];
    if (outputTensor == null) {
      outputTensor =
          outputTensors[index] =
              TensorImpl.fromIndex(
                  interpreterHandle, getOutputTensorIndex(interpreterHandle, index));
    }
    return outputTensor;
  }

  /**
   * Gets the output {@link TensorImpl} given the tensor name and method in the signature.
   *
   * @throws IllegalArgumentException if the output name is invalid.
   */
  TensorImpl getOutputTensor(String outputName, String signatureKey) {
    if (outputName == null) {
      throw new IllegalArgumentException("Invalid output tensor name provided (null)");
    }
    NativeSignatureRunnerWrapper signatureRunnerWrapper = getSignatureRunnerWrapper(signatureKey);
    int subgraphIndex = signatureRunnerWrapper.getSubgraphIndex();
    if (subgraphIndex > 0) {
      return signatureRunnerWrapper.getOutputTensor(outputName);
    }

    int outputIndex = signatureRunnerWrapper.getOutputIndex(outputName);
    return getOutputTensor(outputIndex);
  }

  /** Gets the number of ops in the execution plan. */
  int getExecutionPlanLength() {
    return getExecutionPlanLength(interpreterHandle);
  }

  /**
   * Sets internal cancellation flag. If it's true, the interpreter will try to interrupt any
   * invocation between ops.
   */
  void setCancelled(boolean value) {
    if (cancellationFlagHandle == 0) {
      throw new IllegalStateException(
          "Cannot cancel the inference. Have you called InterpreterApi.Options.setCancellable?");
    }
    setCancelled(interpreterHandle, cancellationFlagHandle, value);
  }

  // Add all the delegates specified in the options (other than XNNPACK) to this.delegates.
  private void addDelegates(InterpreterImpl.Options options) {
    // First add the flex delegate if necessary. This ensures the graph is fully resolved before
    // applying other delegates.
    if (originalGraphHasUnresolvedFlexOp) {
      Delegate optionalFlexDelegate = maybeCreateFlexDelegate(options.delegates);
      if (optionalFlexDelegate != null) {
        ownedDelegates.add((AutoCloseable) optionalFlexDelegate);
        delegates.add(optionalFlexDelegate);
      }
    }
    // Now add the user-supplied delegates.
    delegates.addAll(options.delegates);
    if (options.useNNAPI != null && options.useNNAPI.booleanValue()) {
      NnApiDelegate optionalNnApiDelegate = new NnApiDelegate();
      ownedDelegates.add(optionalNnApiDelegate);
      delegates.add(optionalNnApiDelegate);
    }
    // Finally add the XNNPACK delegate if enabled.
    maybeAddXnnpackDelegate(options);
  }

  // Optionally add the XNNPACK delegate.
  private void maybeAddXnnpackDelegate(InterpreterImpl.Options options) {
    // Simply use "-1" to represent the default mode.
    int applyXNNPACKMode = -1;
    if (options.useXNNPACK != null) {
      applyXNNPACKMode = options.useXNNPACK.booleanValue() ? 1 : 0;
    }

    // TODO(b/171856982): uncomment the following when applying XNNPACK delegate by default is
    // enabled for C++ TfLite library on Android platform.
    if (applyXNNPACKMode == 1 /*|| applyXNNPACKMode == -1*/) {
      XnnpackDelegate xnnpackDelegate =
          createXNNPACKDelegate(
              interpreterHandle, errorHandle, applyXNNPACKMode, options.numThreads);
      delegates.add(xnnpackDelegate);
    }
  }

  // Apply all the delegates specified in this.delegates.
  private void applyDelegates(InterpreterImpl.Options options) {
    for (Delegate delegate : delegates) {
      applyDelegate(interpreterHandle, errorHandle, delegate.getNativeHandle());
    }
  }

  private NativeSignatureRunnerWrapper getSignatureRunnerWrapper(String signatureKey) {
    if (signatureRunnerMap == null) {
      signatureRunnerMap = new HashMap<>();
    }
    if (!signatureRunnerMap.containsKey(signatureKey)) {
      signatureRunnerMap.put(
          signatureKey,
          new NativeSignatureRunnerWrapper(interpreterHandle, errorHandle, signatureKey));
    }
    return signatureRunnerMap.get(signatureKey);
  }

  private static Delegate maybeCreateFlexDelegate(List<Delegate> delegates) {
    try {
      Class<?> clazz = Class.forName("org.tensorflow.lite.flex.FlexDelegate");
      // No need to create the Flex delegate if one has already been provided.
      for (Delegate delegate : delegates) {
        if (clazz.isInstance(delegate)) {
          return null;
        }
      }
      return (Delegate) clazz.getConstructor().newInstance();
    } catch (Exception e) {
      // The error will propagate when tensors are allocated.
      return null;
    }
  }

  private static final int ERROR_BUFFER_SIZE = 512;

  long errorHandle;

  long interpreterHandle;

  private long modelHandle;

  private long cancellationFlagHandle = 0;

  private long inferenceDurationNanoseconds = -1;

  private ByteBuffer modelByteBuffer;

  // Lazily constructed maps of input and output names to input and output Tensor indexes.
  private Map<String, Integer> inputsIndexes;
  private Map<String, Integer> outputsIndexes;
  // Lazily constructed maps of tensor index to index in input and output indexes.
  private Map<Integer, Integer> tensorToInputsIndexes;
  private Map<Integer, Integer> tensorToOutputsIndexes;

  // A map from signature key to its native wrapper object.
  private Map<String, NativeSignatureRunnerWrapper> signatureRunnerMap;

  // Lazily constructed and populated arrays of input and output Tensor wrappers.
  private TensorImpl[] inputTensors;
  private TensorImpl[] outputTensors;

  // Whether subgraph's tensor memory space is allocated.
  private boolean isMemoryAllocated = false;

  // Whether the model has any Flex custom ops that can't be resolved by the OpResolver.
  private boolean originalGraphHasUnresolvedFlexOp = false;

  // As the Java Delegate owns the native delegate instance, we keep a strong ref to any injected
  // delegates for safety.
  private final List<Delegate> delegates = new ArrayList<>();

  // List of owned delegates that must be closed when the interpreter is closed.
  private final List<AutoCloseable> ownedDelegates = new ArrayList<>();

  private static native void run(long interpreterHandle, long errorHandle);

  private static native boolean resizeInput(
      long interpreterHandle, long errorHandle, int inputIdx, int[] dims, boolean strict);

  private static native long allocateTensors(long interpreterHandle, long errorHandle);

  private static native String[] getSignatureKeys(long interpreterHandle);

  private static native void setCancelled(
      long interpreterHandle, long cancellationFlagHandle, boolean value);

  private static native int getOutputDataType(long interpreterHandle, int outputIdx);

  private static native boolean hasUnresolvedFlexOp(long interpreterHandle);

  private static native int getInputTensorIndex(long interpreterHandle, int inputIdx);

  private static native int getOutputTensorIndex(long interpreterHandle, int outputIdx);

  private static native int getInputCount(long interpreterHandle);

  private static native int getOutputCount(long interpreterHandle);

  private static native int getExecutionPlanLength(long interpreterHandle);

  private static native String[] getInputNames(long interpreterHandle);

  private static native String[] getOutputNames(long interpreterHandle);

  private static native void allowFp16PrecisionForFp32(long interpreterHandle, boolean allow);

  private static native void allowBufferHandleOutput(long interpreterHandle, boolean allow);

  private static native XnnpackDelegate createXNNPACKDelegate(
      long interpreterHandle, long errorHandle, int state, int numThreads);

  private static native long createErrorReporter(int size);

  private static native long createModel(String modelPathOrBuffer, long errorHandle);

  private static native long createModelWithBuffer(ByteBuffer modelBuffer, long errorHandle);

  private static native long createInterpreter(long modelHandle, long errorHandle, int numThreads);

  private static native void applyDelegate(
      long interpreterHandle, long errorHandle, long delegateHandle);

  private static native void resetVariableTensors(long interpreterHandle, long errorHandle);

  private static native long createCancellationFlag(long interpreterHandle);

  private static native long deleteCancellationFlag(long cancellationFlagHandle);

  private static native void delete(long errorHandle, long modelHandle, long interpreterHandle);
}
