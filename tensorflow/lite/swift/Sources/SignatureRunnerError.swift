// Copyright 2022 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation

/// Errors thrown by the TensorFlow Lite `SignatureRunner`.
public enum SignatureRunnerError: Error, Equatable, Hashable {
  case invalidTensorDataCount(provided: Int, required: Int)
  case invalidTensorDataType
  case failedToCreateSignatureRunner(signatureKey: String)
  case failedToGetTensor(tensorType: String, nameInSignature: String)
  case failedToResizeInputTensor(inputName: String)
  case failedToCopyDataToInputTensor
  case failedToAllocateTensors
  case failedToInvokeSignature(signatureKey: String)
  case allocateTensorsRequired
}

extension SignatureRunnerError: LocalizedError {
  /// A localized description of the signature runner error.
  public var errorDescription: String? {
    switch self {
    case .invalidTensorDataCount(let provided, let required):
      return "Provided data count \(provided) must match the required count \(required)."
    case .invalidTensorDataType:
      return "Tensor data type is unsupported or could not be determined due to a model error."
    case .failedToCreateSignatureRunner(let signatureKey):
      return "Failed to create a signature runner. Signature with key (\(signatureKey)) not found."
    case .failedToGetTensor(let tensorType, let nameInSignature):
      return "Failed to get \(tensorType) tensor with \(tensorType) name (\(nameInSignature))."
    case .failedToResizeInputTensor(let inputName):
      return "Failed to resize input tensor with input name (\(inputName))."
    case .failedToCopyDataToInputTensor:
      return "Failed to copy data to input tensor."
    case .failedToAllocateTensors:
      return "Failed to allocate memory for input tensors."
    case .failedToInvokeSignature(let signatureKey):
      return "Failed to invoke the signature runner with key (\(signatureKey))."
    case .allocateTensorsRequired:
      return "Must call allocateTensors()."
    }
  }
}

extension SignatureRunnerError: CustomStringConvertible {
  /// A textual representation of the TensorFlow Lite signature runner error.
  public var description: String { return errorDescription ?? "Unknown error." }
}
