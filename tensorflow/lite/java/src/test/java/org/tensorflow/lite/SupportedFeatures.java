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
 * This class is used to conditionally enable certain tests that depend on experimental or
 * deprecated features of TF Lite that may not be univerally portable to all TF Lite
 * implementations.
 */
public final class SupportedFeatures {

  private SupportedFeatures() {}

  /**
   * True if the TF Lite implementation supports cancellation.
   *
   * @see Interpreter#setCanceled
   */
  public static native boolean supportsCancellation();

  /**
   * True if the TF Lite implementation supports the XNNPACK delegate.
   *
   * @see Interpreter#setUseXNNPACK
   */
  public static native boolean supportsXnnpack();

  /**
   * True if the TF Lite implementation supports using reduced 16-bit floating point precision for
   * operations that are specified as 32-bit in the model.
   *
   * @see Interpreter#setAllowFp16PrecisionForFp32
   */
  public static native boolean supportsAllowFp16PrecisionForFp32();

  /**
   * True if the TF Lite implementation supports SignatureDef related methods.
   *
   * @see Interpreter#runSignature
   * @see Interpreter#getSignatureKeys
   * @see Interpreter#getSignatureInputs
   * @see Interpreter#getSignatureOutputs
   * @see Interpreter#getInputTensorFromSignature
   * @see Interpreter#getOutputTensorFromSignature
   */
  public static native boolean supportsSignatures();
}
