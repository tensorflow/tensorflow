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
import TensorFlowLiteC

#if os(Linux)
  import SwiftGlibc
#else
  import Darwin
#endif

/// A TensorFlow Lite model signature runner. You can get a `SignatureRunner` instance for a
/// signature from an `Interpreter` and then use the SignatureRunner APIs.
///
/// - Note: `SignatureRunner` instances are *not* thread-safe.
/// - Note: Each `SignatureRunner` instance is associated with an `Interpreter` instance. As long
///     as a `SignatureRunner` instance is still in use, its associated `Interpreter` instance
///     will not be deallocated.
public final class SignatureRunner {
  /// The signature key.
  public let signatureKey: String

  /// The SignatureDefs input names.
  public var inputs: [String] {
    guard let inputs = _inputs else {
      let inputCount = Int(TfLiteSignatureRunnerGetInputCount(self.cSignatureRunner))
      let ins: [String] = (0..<inputCount).map {
        guard
          let inputNameCString = TfLiteSignatureRunnerGetInputName(
            self.cSignatureRunner, Int32($0))
        else {
          return ""
        }
        return String(cString: inputNameCString)
      }
      _inputs = ins
      return ins
    }
    return inputs
  }

  /// The SignatureDefs output names.
  public var outputs: [String] {
    guard let outputs = _outputs else {
      let outputCount = Int(TfLiteSignatureRunnerGetOutputCount(self.cSignatureRunner))
      let outs: [String] = (0..<outputCount).map {
        guard
          let outputNameCString = TfLiteSignatureRunnerGetOutputName(
            self.cSignatureRunner, Int32($0))
        else {
          return ""
        }
        return String(cString: outputNameCString)
      }
      _outputs = outs
      return outs
    }
    return outputs
  }

  /// The backing interpreter. It's a strong reference to ensure that the interpreter is never
  /// released before this signature runner is released.
  ///
  /// - Warning: Never let the interpreter hold a strong reference to the signature runner to avoid
  ///     retain cycles.
  private var interpreter: Interpreter

  /// The `TfLiteSignatureRunner` C pointer type represented as an
  /// `UnsafePointer<TfLiteSignatureRunner>`.
  private typealias CSignatureRunner = OpaquePointer
  /// The `TfLiteTensor` C pointer type represented as an
  /// `UnsafePointer<TfLiteTensor>`.
  private typealias CTensor = UnsafePointer<TfLiteTensor>?

  /// The underlying `TfLiteSignatureRunner` C pointer.
  private var cSignatureRunner: CSignatureRunner

  /// Whether we need to allocate tensors memory.
  private var isTensorsAllocationNeeded: Bool = true

  /// The SignatureDefs input names.
  private var _inputs: [String]?

  /// The SignatureDefs output names.
  private var _outputs: [String]?

  // MARK: Initializers

  /// Initializes a new TensorFlow Lite signature runner instance with the given interpreter and
  /// signature key.
  ///
  /// - Parameters:
  ///   - interpreter: The TensorFlow Lite model interpreter.
  ///   - signatureKey: The signature key.
  /// - Throws: An error if fail to create the signature runner with given key.
  internal init(interpreter: Interpreter, signatureKey: String) throws {
    guard let signatureKeyCString = signatureKey.cString(using: String.Encoding.utf8),
      let cSignatureRunner = TfLiteInterpreterGetSignatureRunner(
        interpreter.cInterpreter, signatureKeyCString)
    else {
      throw SignatureRunnerError.failedToCreateSignatureRunner(signatureKey: signatureKey)
    }
    self.cSignatureRunner = cSignatureRunner
    self.signatureKey = signatureKey
    self.interpreter = interpreter
    try allocateTensors()
  }

  deinit {
    TfLiteSignatureRunnerDelete(cSignatureRunner)
  }

  // MARK: Public

  /// Invokes the signature with given input data.
  ///
  /// - Parameters:
  ///   - inputs: A map from input name to the input data. The input data will be copied into the
  ///       input tensor.
  /// - Throws: `SignatureRunnerError` if input data copying or signature invocation fails.
  public func invoke(with inputs: [String: Data]) throws {
    try allocateTensors()
    for (inputName, inputData) in inputs {
      try copy(inputData, toInputNamed: inputName)
    }
    guard TfLiteSignatureRunnerInvoke(self.cSignatureRunner) == kTfLiteOk else {
      throw SignatureRunnerError.failedToInvokeSignature(signatureKey: signatureKey)
    }
  }

  /// Returns the input tensor with the given input name in the signature.
  ///
  /// - Parameters:
  ///   - name: The input name in the signature.
  /// - Throws: An error if fail to get the input `Tensor` or the `Tensor` is invalid.
  /// - Returns: The input `Tensor` with the given input name.
  public func input(named name: String) throws -> Tensor {
    return try tensor(named: name, withType: TensorType.input)
  }

  /// Returns the output tensor with the given output name in the signature.
  ///
  /// - Parameters:
  ///   - name: The output name in the signature.
  /// - Throws: An error if fail to get the output `Tensor` or the `Tensor` is invalid.
  /// - Returns: The output `Tensor` with the given output name.
  public func output(named name: String) throws -> Tensor {
    return try tensor(named: name, withType: TensorType.output)
  }

  /// Resizes the input `Tensor` with the given input name to the specified `Tensor.Shape`.
  ///
  /// - Note: After resizing an input tensor, the client **must** explicitly call
  ///     `allocateTensors()` before attempting to access the resized tensor data.
  /// - Parameters:
  ///   - name: The input name of the `Tensor`.
  ///   - shape: The shape to resize the input `Tensor` to.
  /// - Throws: An error if the input tensor with given input name could not be resized.
  public func resizeInput(named name: String, toShape shape: Tensor.Shape) throws {
    guard let inputNameCString = name.cString(using: String.Encoding.utf8),
      TfLiteSignatureRunnerResizeInputTensor(
        self.cSignatureRunner,
        inputNameCString,
        shape.int32Dimensions,
        Int32(shape.rank)
      ) == kTfLiteOk
    else {
      throw SignatureRunnerError.failedToResizeInputTensor(inputName: name)
    }
    isTensorsAllocationNeeded = true
  }

  /// Copies the given data to the input `Tensor` with the given input name.
  ///
  /// - Parameters:
  ///   - data: The data to be copied to the input `Tensor`'s data buffer.
  ///   - name: The input name of the `Tensor`.
  /// - Throws: An error if fail to get the input `Tensor` or if the `data.count` does not match the
  ///     input tensor's `data.count`.
  /// - Returns: The input `Tensor` with the copied data.
  public func copy(_ data: Data, toInputNamed name: String) throws {
    guard let inputNameCString = name.cString(using: String.Encoding.utf8),
      let cTensor = TfLiteSignatureRunnerGetInputTensor(self.cSignatureRunner, inputNameCString)
    else {
      throw SignatureRunnerError.failedToGetTensor(tensorType: "input", nameInSignature: name)
    }

    let byteCount = TfLiteTensorByteSize(cTensor)
    guard data.count == byteCount else {
      throw SignatureRunnerError.invalidTensorDataCount(provided: data.count, required: byteCount)
    }

    #if swift(>=5.0)
      let status = data.withUnsafeBytes {
        TfLiteTensorCopyFromBuffer(cTensor, $0.baseAddress, data.count)
      }
    #else
      let status = data.withUnsafeBytes { TfLiteTensorCopyFromBuffer(cTensor, $0, data.count) }
    #endif  // swift(>=5.0)
    guard status == kTfLiteOk else { throw SignatureRunnerError.failedToCopyDataToInputTensor }
  }

  /// Allocates memory for tensors.
  /// - Note: This is a relatively expensive operation and this call is *purely optional*.
  ///     Tensor allocation will occur automatically during execution.
  /// - Throws: An error if memory could not be allocated for the tensors.
  public func allocateTensors() throws {
    if !isTensorsAllocationNeeded { return }
    guard TfLiteSignatureRunnerAllocateTensors(self.cSignatureRunner) == kTfLiteOk else {
      throw SignatureRunnerError.failedToAllocateTensors
    }
    isTensorsAllocationNeeded = false
  }

  // MARK: - Private

  /// Returns the I/O tensor with the given name in the signature.
  ///
  /// - Parameters:
  ///   - nameInSignature: The input or output name in the signature.
  ///   - type: The tensor type.
  /// - Throws: An error if fail to get the `Tensor` or the `Tensor` is invalid.
  /// - Returns: The `Tensor` with the given name in the signature.
  private func tensor(named nameInSignature: String, withType type: TensorType) throws -> Tensor {
    guard let nameInSignatureCString = nameInSignature.cString(using: String.Encoding.utf8)
    else {
      throw SignatureRunnerError.failedToGetTensor(
        tensorType: type.rawValue, nameInSignature: nameInSignature)
    }
    var cTensorPointer: CTensor
    switch type {
    case .input:
      cTensorPointer = UnsafePointer(
        TfLiteSignatureRunnerGetInputTensor(self.cSignatureRunner, nameInSignatureCString))
    case .output:
      cTensorPointer = TfLiteSignatureRunnerGetOutputTensor(
        self.cSignatureRunner, nameInSignatureCString)
    }
    guard let cTensor = cTensorPointer else {
      throw SignatureRunnerError.failedToGetTensor(
        tensorType: type.rawValue, nameInSignature: nameInSignature)
    }
    guard let bytes = TfLiteTensorData(cTensor) else {
      throw SignatureRunnerError.allocateTensorsRequired
    }
    guard let dataType = Tensor.DataType(type: TfLiteTensorType(cTensor)) else {
      throw SignatureRunnerError.invalidTensorDataType
    }
    let nameCString = TfLiteTensorName(cTensor)
    let name = nameCString == nil ? "" : String(cString: nameCString!)
    let byteCount = TfLiteTensorByteSize(cTensor)
    let data = Data(bytes: bytes, count: byteCount)
    let rank = TfLiteTensorNumDims(cTensor)
    let dimensions = (0..<rank).map { Int(TfLiteTensorDim(cTensor, $0)) }
    let shape = Tensor.Shape(dimensions)
    let cQuantizationParams = TfLiteTensorQuantizationParams(cTensor)
    let scale = cQuantizationParams.scale
    let zeroPoint = Int(cQuantizationParams.zero_point)
    var quantizationParameters: QuantizationParameters? = nil
    if scale != 0.0 {
      quantizationParameters = QuantizationParameters(scale: scale, zeroPoint: zeroPoint)
    }
    let tensor = Tensor(
      name: name,
      dataType: dataType,
      shape: shape,
      data: data,
      quantizationParameters: quantizationParameters
    )
    return tensor
  }
}
