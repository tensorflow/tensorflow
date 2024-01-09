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
import TensorFlowLiteC

#if os(Linux)
  import SwiftGlibc
#else
  import Darwin
#endif

/// A TensorFlow Lite interpreter that performs inference from a given model.
///
/// - Note: Interpreter instances are *not* thread-safe.
public final class Interpreter {
  /// The configuration options for the `Interpreter`.
  public let options: Options?

  /// An `Array` of `Delegate`s for the `Interpreter` to use to perform graph operations.
  public let delegates: [Delegate]?

  /// The total number of input `Tensor`s associated with the model.
  public var inputTensorCount: Int {
    return Int(TfLiteInterpreterGetInputTensorCount(cInterpreter))
  }

  /// The total number of output `Tensor`s associated with the model.
  public var outputTensorCount: Int {
    return Int(TfLiteInterpreterGetOutputTensorCount(cInterpreter))
  }

  /// An ordered list of SignatureDef exported method names available in the model.
  public var signatureKeys: [String] {
    guard let signatureKeys = _signatureKeys else {
      let signatureCount = Int(TfLiteInterpreterGetSignatureCount(self.cInterpreter))
      let keys: [String] = (0..<signatureCount).map {
        guard
          let signatureNameCString = TfLiteInterpreterGetSignatureKey(
            self.cInterpreter, Int32($0))
        else {
          return ""
        }
        return String(cString: signatureNameCString)
      }
      _signatureKeys = keys
      return keys
    }
    return signatureKeys
  }

  /// The `TfLiteInterpreter` C pointer type represented as an `UnsafePointer<TfLiteInterpreter>`.
  internal typealias CInterpreter = OpaquePointer

  /// The underlying `TfLiteInterpreter` C pointer.
  internal var cInterpreter: CInterpreter?

  /// The underlying `TfLiteDelegate` C pointer for XNNPACK delegate.
  private var cXNNPackDelegate: Delegate.CDelegate?

  /// An ordered list of SignatureDef exported method names available in the model.
  private var _signatureKeys: [String]? = nil

  /// Creates a new instance with the given values.
  ///
  /// - Parameters:
  ///   - modelPath: The local file path to a TensorFlow Lite model.
  ///   - options: Configurations for the `Interpreter`. The default is `nil` indicating that the
  ///       `Interpreter` will determine the configuration options.
  ///   - delegate: `Array` of `Delegate`s for the `Interpreter` to use to peform graph operations.
  ///       The default is `nil`.
  /// - Throws: An error if the model could not be loaded or the interpreter could not be created.
  public convenience init(modelPath: String, options: Options? = nil, delegates: [Delegate]? = nil)
    throws
  {
    guard let model = Model(filePath: modelPath) else { throw InterpreterError.failedToLoadModel }
    try self.init(model: model, options: options, delegates: delegates)
  }

  /// Creates a new instance with the given values.
  ///
  /// - Parameters:
  ///   - modelData: Binary data representing a TensorFlow Lite model.
  ///   - options: Configurations for the `Interpreter`. The default is `nil` indicating that the
  ///       `Interpreter` will determine the configuration options.
  ///   - delegate: `Array` of `Delegate`s for the `Interpreter` to use to peform graph operations.
  ///       The default is `nil`.
  /// - Throws: An error if the model could not be loaded or the interpreter could not be created.
  public convenience init(modelData: Data, options: Options? = nil, delegates: [Delegate]? = nil)
    throws
  {
    guard let model = Model(modelData: modelData) else { throw InterpreterError.failedToLoadModel }
    try self.init(model: model, options: options, delegates: delegates)
  }

  /// Create a new instance with the given values.
  ///
  /// - Parameters:
  ///   - model: An instantiated TensorFlow Lite model.
  ///   - options: Configurations for the `Interpreter`. The default is `nil` indicating that the
  ///       `Interpreter` will determine the configuration options.
  ///   - delegate: `Array` of `Delegate`s for the `Interpreter` to use to peform graph operations.
  ///       The default is `nil`.
  /// - Throws: An error if the model could not be loaded or the interpreter could not be created.
  private init(model: Model, options: Options? = nil, delegates: [Delegate]? = nil) throws {
    guard let cInterpreterOptions = TfLiteInterpreterOptionsCreate() else {
      throw InterpreterError.failedToCreateInterpreter
    }
    defer { TfLiteInterpreterOptionsDelete(cInterpreterOptions) }

    self.options = options
    self.delegates = delegates
    options.map {
      if let threadCount = $0.threadCount, threadCount > 0 {
        TfLiteInterpreterOptionsSetNumThreads(cInterpreterOptions, Int32(threadCount))
      }
      TfLiteInterpreterOptionsSetErrorReporter(
        cInterpreterOptions,
        { (_, format, args) -> Void in
          // Workaround for optionality differences for x86_64 (non-optional) and arm64 (optional).
          let optionalArgs: CVaListPointer? = args
          guard let cFormat = format,
            let arguments = optionalArgs,
            let message = String(cFormat: cFormat, arguments: arguments)
          else {
            return
          }
          print(String(describing: InterpreterError.tensorFlowLiteError(message)))
        },
        nil
      )
    }
    delegates?.forEach { TfLiteInterpreterOptionsAddDelegate(cInterpreterOptions, $0.cDelegate) }

    // Configure the XNNPack delegate after the other delegates explicitly added by the user.
    options.map {
      if $0.isXNNPackEnabled {
        configureXNNPack(options: $0, cInterpreterOptions: cInterpreterOptions)
      }
    }

    guard let cInterpreter = TfLiteInterpreterCreate(model.cModel, cInterpreterOptions) else {
      throw InterpreterError.failedToCreateInterpreter
    }
    self.cInterpreter = cInterpreter
  }

  deinit {
    TfLiteInterpreterDelete(cInterpreter)
    TfLiteXNNPackDelegateDelete(cXNNPackDelegate)
  }

  /// Invokes the interpreter to perform inference from the loaded graph.
  ///
  /// - Throws: An error if the model was not ready because the tensors were not allocated.
  public func invoke() throws {
    guard TfLiteInterpreterInvoke(cInterpreter) == kTfLiteOk else {
      throw InterpreterError.allocateTensorsRequired
    }
  }

  /// Returns the input `Tensor` at the given index.
  ///
  /// - Parameters:
  ///   - index: The index for the input `Tensor`.
  /// - Throws: An error if the index is invalid or the tensors have not been allocated.
  /// - Returns: The input `Tensor` at the given index.
  public func input(at index: Int) throws -> Tensor {
    let maxIndex = inputTensorCount - 1
    guard case 0...maxIndex = index else {
      throw InterpreterError.invalidTensorIndex(index: index, maxIndex: maxIndex)
    }
    guard let cTensor = TfLiteInterpreterGetInputTensor(cInterpreter, Int32(index)),
      let bytes = TfLiteTensorData(cTensor),
      let nameCString = TfLiteTensorName(cTensor)
    else {
      throw InterpreterError.allocateTensorsRequired
    }
    guard let dataType = Tensor.DataType(type: TfLiteTensorType(cTensor)) else {
      throw InterpreterError.invalidTensorDataType
    }

    let name = String(cString: nameCString)
    let rank = TfLiteTensorNumDims(cTensor)
    let dimensions = (0..<rank).map { Int(TfLiteTensorDim(cTensor, $0)) }
    let shape = Tensor.Shape(dimensions)
    let byteCount = TfLiteTensorByteSize(cTensor)
    let data = Data(bytes: bytes, count: byteCount)
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

  /// Returns the output `Tensor` at the given index.
  ///
  /// - Parameters:
  ///   - index: The index for the output `Tensor`.
  /// - Throws: An error if the index is invalid, tensors haven't been allocated, or interpreter
  ///     has not been invoked for models that dynamically compute output tensors based on the
  ///     values of its input tensors.
  /// - Returns: The output `Tensor` at the given index.
  public func output(at index: Int) throws -> Tensor {
    let maxIndex = outputTensorCount - 1
    guard case 0...maxIndex = index else {
      throw InterpreterError.invalidTensorIndex(index: index, maxIndex: maxIndex)
    }
    guard let cTensor = TfLiteInterpreterGetOutputTensor(cInterpreter, Int32(index)),
      let bytes = TfLiteTensorData(cTensor),
      let nameCString = TfLiteTensorName(cTensor)
    else {
      throw InterpreterError.invokeInterpreterRequired
    }
    guard let dataType = Tensor.DataType(type: TfLiteTensorType(cTensor)) else {
      throw InterpreterError.invalidTensorDataType
    }

    let name = String(cString: nameCString)
    let rank = TfLiteTensorNumDims(cTensor)
    let dimensions = (0..<rank).map { Int(TfLiteTensorDim(cTensor, $0)) }
    let shape = Tensor.Shape(dimensions)
    let byteCount = TfLiteTensorByteSize(cTensor)
    let data = Data(bytes: bytes, count: byteCount)
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

  /// Resizes the input `Tensor` at the given index to the specified `Tensor.Shape`.
  ///
  /// - Note: After resizing an input tensor, the client **must** explicitly call
  ///     `allocateTensors()` before attempting to access the resized tensor data or invoking the
  ///     interpreter to perform inference.
  /// - Parameters:
  ///   - index: The index for the input `Tensor`.
  ///   - shape: The shape to resize the input `Tensor` to.
  /// - Throws: An error if the input tensor at the given index could not be resized.
  public func resizeInput(at index: Int, to shape: Tensor.Shape) throws {
    let maxIndex = inputTensorCount - 1
    guard case 0...maxIndex = index else {
      throw InterpreterError.invalidTensorIndex(index: index, maxIndex: maxIndex)
    }
    guard
      TfLiteInterpreterResizeInputTensor(
        cInterpreter,
        Int32(index),
        shape.int32Dimensions,
        Int32(shape.rank)
      ) == kTfLiteOk
    else {
      throw InterpreterError.failedToResizeInputTensor(index: index)
    }
  }

  /// Copies the given data to the input `Tensor` at the given index.
  ///
  /// - Parameters:
  ///   - data: The data to be copied to the input `Tensor`'s data buffer.
  ///   - index: The index for the input `Tensor`.
  /// - Throws: An error if the `data.count` does not match the input tensor's `data.count` or if
  ///     the given index is invalid.
  /// - Returns: The input `Tensor` with the copied data.
  @discardableResult
  public func copy(_ data: Data, toInputAt index: Int) throws -> Tensor {
    let maxIndex = inputTensorCount - 1
    guard case 0...maxIndex = index else {
      throw InterpreterError.invalidTensorIndex(index: index, maxIndex: maxIndex)
    }
    guard let cTensor = TfLiteInterpreterGetInputTensor(cInterpreter, Int32(index)) else {
      throw InterpreterError.allocateTensorsRequired
    }

    let byteCount = TfLiteTensorByteSize(cTensor)
    guard data.count == byteCount else {
      throw InterpreterError.invalidTensorDataCount(provided: data.count, required: byteCount)
    }

    #if swift(>=5.0)
      let status = data.withUnsafeBytes {
        TfLiteTensorCopyFromBuffer(cTensor, $0.baseAddress, data.count)
      }
    #else
      let status = data.withUnsafeBytes { TfLiteTensorCopyFromBuffer(cTensor, $0, data.count) }
    #endif  // swift(>=5.0)
    guard status == kTfLiteOk else { throw InterpreterError.failedToCopyDataToInputTensor }
    return try input(at: index)
  }

  /// Allocates memory for all input `Tensor`s based on their `Tensor.Shape`s.
  ///
  /// - Note: This is a relatively expensive operation and should only be called after creating the
  ///     interpreter and resizing any input tensors.
  /// - Throws: An error if memory could not be allocated for the input tensors.
  public func allocateTensors() throws {
    guard TfLiteInterpreterAllocateTensors(cInterpreter) == kTfLiteOk else {
      throw InterpreterError.failedToAllocateTensors
    }
  }

  /// Returns a new signature runner instance for the signature with the given key in the model.
  ///
  /// - Parameters:
  ///   - key: The signature key.
  /// - Throws: `SignatureRunnerError` if signature runner creation fails.
  /// - Returns: A new signature runner instance for the signature with the given key.
  public func signatureRunner(with key: String) throws -> SignatureRunner {
    guard signatureKeys.contains(key) else {
      throw SignatureRunnerError.failedToCreateSignatureRunner(signatureKey: key)
    }
    return try SignatureRunner.init(interpreter: self, signatureKey: key)
  }

  // MARK: - Private

  private func configureXNNPack(options: Options, cInterpreterOptions: OpaquePointer) {
    var cXNNPackOptions = TfLiteXNNPackDelegateOptionsDefault()
    if let threadCount = options.threadCount, threadCount > 0 {
      cXNNPackOptions.num_threads = Int32(threadCount)
    }

    cXNNPackDelegate = TfLiteXNNPackDelegateCreate(&cXNNPackOptions)
    TfLiteInterpreterOptionsAddDelegate(cInterpreterOptions, cXNNPackDelegate)
  }
}

extension Interpreter {
  /// Options for configuring the `Interpreter`.
  public struct Options: Equatable, Hashable {
    /// The maximum number of CPU threads that the interpreter should run on. The default is `nil`
    /// indicating that the `Interpreter` will decide the number of threads to use.
    public var threadCount: Int? = nil

    /// Indicates whether an optimized set of floating point CPU kernels, provided by XNNPACK, is
    /// enabled.
    ///
    /// - Experiment:
    /// Enabling this flag will enable use of a new, highly optimized set of CPU kernels provided
    /// via the XNNPACK delegate. Currently, this is restricted to a subset of floating point
    /// operations. Eventually, we plan to enable this by default, as it can provide significant
    /// performance benefits for many classes of floating point models. See
    /// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/xnnpack/README.md
    /// for more details.
    ///
    /// - Important:
    /// Things to keep in mind when enabling this flag:
    ///
    ///     * Startup time and resize time may increase.
    ///     * Baseline memory consumption may increase.
    ///     * Compatibility with other delegates (e.g., GPU) has not been fully validated.
    ///     * Quantized models will not see any benefit.
    ///
    /// - Warning: This is an experimental interface that is subject to change.
    public var isXNNPackEnabled: Bool = false

    /// Creates a new instance with the default values.
    public init() {}
  }
}

/// A type alias for `Interpreter.Options` to support backwards compatibility with the deprecated
/// `InterpreterOptions` struct.
@available(*, deprecated, renamed: "Interpreter.Options")
public typealias InterpreterOptions = Interpreter.Options

extension String {
  /// Returns a new `String` initialized by using the given format C array as a template into which
  /// the remaining argument values are substituted according to the user’s default locale.
  ///
  /// - Note: Returns `nil` if a new `String` could not be constructed from the given values.
  /// - Parameters:
  ///   - cFormat: The format C array as a template for substituting values.
  ///   - arguments: A C pointer to a `va_list` of arguments to substitute into `cFormat`.
  init?(cFormat: UnsafePointer<CChar>, arguments: CVaListPointer) {
    #if os(Linux)
      let length = Int(vsnprintf(nil, 0, cFormat, arguments) + 1)  // null terminator
      guard length > 0 else { return nil }
      let buffer = UnsafeMutablePointer<CChar>.allocate(capacity: length)
      defer {
        buffer.deallocate()
      }
      guard vsnprintf(buffer, length, cFormat, arguments) == length - 1 else { return nil }
      self.init(validatingUTF8: buffer)
    #else
      var buffer: UnsafeMutablePointer<CChar>?
      guard vasprintf(&buffer, cFormat, arguments) != 0, let cString = buffer else { return nil }
      self.init(validatingUTF8: cString)
    #endif
  }
}
