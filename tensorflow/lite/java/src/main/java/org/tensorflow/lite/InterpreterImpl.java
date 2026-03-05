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
import java.util.HashMap;
import java.util.Map;
import org.checkerframework.checker.nullness.qual.NonNull;

/**
 * Package-private class that implements InterpreterApi. This class implements all the
 * non-experimental API methods. It is used both by the public InterpreterFactory, and as a base
 * class for the public Interpreter class,
 */
class InterpreterImpl implements InterpreterApi {
  /**
   * An options class for controlling runtime interpreter behavior. Compared to the base class
   * InterpreterApi.Options, this adds fields corresponding to experimental features. But it does
   * not provide accessors to set those fields -- those are only provided in the derived class
   * Interpreter.Options.
   */
  static class Options extends InterpreterApi.Options {
    public Options() {}

    public Options(InterpreterApi.Options options) {
      super(options);
    }

    public Options(Options other) {
      super(other);
      allowFp16PrecisionForFp32 = other.allowFp16PrecisionForFp32;
      allowBufferHandleOutput = other.allowBufferHandleOutput;
    }

    // See Interpreter.Options#setAllowFp16PrecisionForFp32(boolean).
    Boolean allowFp16PrecisionForFp32;

    // See Interpreter.Options#setAllowBufferHandleOutput(boolean).
    Boolean allowBufferHandleOutput;
  }

  /**
   * Initializes an {@code InterpreterImpl} and specifies options for customizing interpreter
   * behavior.
   *
   * @param modelFile a file of a pre-trained TF Lite model
   * @param options a set of options for customizing interpreter behavior
   * @throws IllegalArgumentException if {@code modelFile} does not encode a valid TensorFlow Lite
   *     model.
   */
  InterpreterImpl(@NonNull File modelFile, Options options) {
    wrapper = new NativeInterpreterWrapper(modelFile.getAbsolutePath(), options);
    signatureKeyList = getSignatureKeys();
  }

  /**
   * Initializes an {@code InterpreterImpl} with a {@code ByteBuffer} of a model file and a set of
   * custom {@link Interpreter.Options}.
   *
   * <p>The {@code ByteBuffer} should not be modified after the construction of an {@code
   * InterpreterImpl}. The {@code ByteBuffer} can be either a {@code MappedByteBuffer} that
   * memory-maps a model file, or a direct {@code ByteBuffer} of nativeOrder() that contains the
   * bytes content of a model.
   *
   * @throws IllegalArgumentException if {@code byteBuffer} is not a {@code MappedByteBuffer} nor a
   *     direct {@code ByteBuffer} of nativeOrder.
   */
  InterpreterImpl(@NonNull ByteBuffer byteBuffer, Options options) {
    wrapper = new NativeInterpreterWrapper(byteBuffer, options);
    signatureKeyList = getSignatureKeys();
  }

  InterpreterImpl(NativeInterpreterWrapper wrapper) {
    this.wrapper = wrapper;
    signatureKeyList = getSignatureKeys();
  }

  @Override
  public void run(Object input, Object output) {
    Object[] inputs = {input};
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, output);
    runForMultipleInputsOutputs(inputs, outputs);
  }

  @Override
  public void runForMultipleInputsOutputs(
      Object @NonNull [] inputs, @NonNull Map<Integer, Object> outputs) {
    checkNotClosed();
    wrapper.run(inputs, outputs);
  }

  @Override
  public void allocateTensors() {
    checkNotClosed();
    wrapper.allocateTensors();
  }

  @Override
  public void resizeInput(int idx, int @NonNull [] dims) {
    checkNotClosed();
    wrapper.resizeInput(idx, dims, false);
  }

  @Override
  public void resizeInput(int idx, int @NonNull [] dims, boolean strict) {
    checkNotClosed();
    wrapper.resizeInput(idx, dims, strict);
  }

  @Override
  public int getInputTensorCount() {
    checkNotClosed();
    return wrapper.getInputTensorCount();
  }

  @Override
  public int getInputIndex(String opName) {
    checkNotClosed();
    return wrapper.getInputIndex(opName);
  }

  @Override
  public Tensor getInputTensor(int inputIndex) {
    checkNotClosed();
    return wrapper.getInputTensor(inputIndex);
  }

  /** Gets the number of output Tensors. */
  @Override
  public int getOutputTensorCount() {
    checkNotClosed();
    return wrapper.getOutputTensorCount();
  }

  @Override
  public int getOutputIndex(String opName) {
    checkNotClosed();
    return wrapper.getOutputIndex(opName);
  }

  @Override
  public Tensor getOutputTensor(int outputIndex) {
    checkNotClosed();
    return wrapper.getOutputTensor(outputIndex);
  }

  @Override
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

  @Override
  public void runSignature(
      @NonNull Map<String, Object> inputs, @NonNull Map<String, Object> outputs) {
    checkNotClosed();
    runSignature(inputs, outputs, null);
  }

  @Override
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

  @Override
  public String[] getSignatureKeys() {
    checkNotClosed();
    return wrapper.getSignatureKeys();
  }

  @Override
  public String[] getSignatureInputs(String signatureKey) {
    checkNotClosed();
    return wrapper.getSignatureInputs(signatureKey);
  }

  @Override
  public String[] getSignatureOutputs(String signatureKey) {
    checkNotClosed();
    return wrapper.getSignatureOutputs(signatureKey);
  }

  @Override
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

  @Override
  public Long getLastNativeInferenceDurationNanoseconds() {
    checkNotClosed();
    return wrapper.getLastNativeInferenceDurationNanoseconds();
  }

  int getExecutionPlanLength() {
    checkNotClosed();
    return wrapper.getExecutionPlanLength();
  }

  @Override
  public void close() {
    if (wrapper != null) {
      wrapper.close();
      wrapper = null;
    }
  }

  @SuppressWarnings("deprecation")
  @Override
  protected void finalize() throws Throwable {
    try {
      close();
    } finally {
      super.finalize();
    }
  }

  void checkNotClosed() {
    if (wrapper == null) {
      throw new IllegalStateException("Internal error: The Interpreter has already been closed.");
    }
  }

  NativeInterpreterWrapper wrapper;
  private final String[] signatureKeyList;
}
