/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

/**
 * An internal wrapper that wraps native SignatureRunner.
 *
 * <p>Note: This class is not thread safe.
 */
final class NativeSignatureRunnerWrapper {
  NativeSignatureRunnerWrapper(long interpreterHandle, long errorHandle, String signatureKey) {
    this.errorHandle = errorHandle;
    signatureRunnerHandle = nativeGetSignatureRunner(interpreterHandle, signatureKey);
    if (signatureRunnerHandle == -1) {
      throw new IllegalArgumentException("Input error: Signature " + signatureKey + " not found.");
    }
  }

  /**
   * Attempts to get the subgraph index associated with this Signature. Returns the subgraph index,
   * or -1 on error.
   */
  public int getSubgraphIndex() {
    return nativeGetSubgraphIndex(signatureRunnerHandle);
  }

  /** Gets the inputs of this Signature. */
  public String[] inputNames() {
    return nativeInputNames(signatureRunnerHandle);
  }

  /** Gets the outputs of this Signature. */
  public String[] outputNames() {
    return nativeOutputNames(signatureRunnerHandle);
  }

  /** Gets the input tensor specified by {@code inputName}. */
  public TensorImpl getInputTensor(String inputName) {
    return TensorImpl.fromSignatureInput(signatureRunnerHandle, inputName);
  }

  /** Gets the output tensor specified by {@code outputName}. */
  public TensorImpl getOutputTensor(String outputName) {
    return TensorImpl.fromSignatureOutput(signatureRunnerHandle, outputName);
  }

  /** Gets the index of the input specified by {@code inputName}. */
  public int getInputIndex(String inputName) {
    int inputIndex = nativeGetInputIndex(signatureRunnerHandle, inputName);
    if (inputIndex == -1) {
      throw new IllegalArgumentException("Input error: input " + inputName + " not found.");
    }
    return inputIndex;
  }

  /** Gets the index of the output specified by {@code outputName}. */
  public int getOutputIndex(String outputName) {
    int outputIndex = nativeGetOutputIndex(signatureRunnerHandle, outputName);
    if (outputIndex == -1) {
      throw new IllegalArgumentException("Input error: output " + outputName + " not found.");
    }
    return outputIndex;
  }

  /** Resizes dimensions of a specific input. */
  public boolean resizeInput(String inputName, int[] dims) {
    isMemoryAllocated = false;
    return nativeResizeInput(signatureRunnerHandle, errorHandle, inputName, dims);
  }

  /** Allocates tensor memory space. */
  public void allocateTensorsIfNeeded() {
    if (isMemoryAllocated) {
      return;
    }

    nativeAllocateTensors(signatureRunnerHandle, errorHandle);
    isMemoryAllocated = true;
  }

  /** Runs inference for this Signature. */
  public void invoke() {
    nativeInvoke(signatureRunnerHandle, errorHandle);
  }

  private final long signatureRunnerHandle;

  private final long errorHandle;

  private boolean isMemoryAllocated = false;

  private static native long nativeGetSignatureRunner(long interpreterHandle, String signatureKey);

  private static native int nativeGetSubgraphIndex(long signatureRunnerHandle);

  private static native String[] nativeInputNames(long signatureRunnerHandle);

  private static native String[] nativeOutputNames(long signatureRunnerHandle);

  private static native int nativeGetInputIndex(long signatureRunnerHandle, String inputName);

  private static native int nativeGetOutputIndex(long signatureRunnerHandle, String outputName);

  private static native boolean nativeResizeInput(
      long signatureRunnerHandle, long errorHandle, String inputName, int[] dims);

  private static native void nativeAllocateTensors(long signatureRunnerHandle, long errorHandle);

  private static native void nativeInvoke(long signatureRunnerHandle, long errorHandle);
}
