// Copyright 2018 Google Inc. All rights reserved.
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

/// Errors thrown by the TensorFlow Lite `Interpreter`.
public enum InterpreterError: Error, Equatable, Hashable {
  case invalidTensorIndex(index: Int, maxIndex: Int)
  case invalidTensorDataCount(provided: Int, required: Int)
  case invalidTensorDataType
  case failedToLoadModel
  case failedToCreateInterpreter
  case failedToResizeInputTensor(index: Int)
  case failedToCopyDataToInputTensor
  case failedToAllocateTensors
  case allocateTensorsRequired
  case invokeInterpreterRequired
  case tensorFlowLiteError(String)
}

extension InterpreterError: LocalizedError {
  /// A localized description of the interpreter error.
  public var errorDescription: String? {
    switch self {
    case .invalidTensorIndex(let index, let maxIndex):
      return "Invalid tensor index \(index), max index is \(maxIndex)."
    case .invalidTensorDataCount(let provided, let required):
      return "Provided data count \(provided) must match the required count \(required)."
    case .invalidTensorDataType:
      return "Tensor data type is unsupported or could not be determined due to a model error."
    case .failedToLoadModel:
      return "Failed to load the given model."
    case .failedToCreateInterpreter:
      return "Failed to create the interpreter."
    case .failedToResizeInputTensor(let index):
      return "Failed to resize input tesnor at index \(index)."
    case .failedToCopyDataToInputTensor:
      return "Failed to copy data to input tensor."
    case .failedToAllocateTensors:
      return "Failed to allocate memory for input tensors."
    case .allocateTensorsRequired:
      return "Must call allocateTensors()."
    case .invokeInterpreterRequired:
      return "Must call invoke()."
    case .tensorFlowLiteError(let message):
      return "TensorFlow Lite Error: \(message)"
    }
  }
}

extension InterpreterError: CustomStringConvertible {
  /// A textual representation of the TensorFlow Lite interpreter error.
  public var description: String { return errorDescription ?? "Unknown error." }
}
