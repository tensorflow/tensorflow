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

/// A TensorFlow Lite interpreter that performs inference from a given model.
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

  /// The `TfLiteInterpreter` C pointer type represented as an `UnsafePointer<TfLiteInterpreter>`.
  private typealias CInterpreter = OpaquePointer

  /// The underlying `TfLiteInterpreter` C pointer.
  private var cInterpreter: CInterpreter?

  /// Creates a new instance with the given values.
  ///
  /// - Parameters:
  ///   - modelPath: The local file path to a TensorFlow Lite model.
  ///   - options: Configurations for the `Interpreter`. The default is `nil` indicating that the
  ///       `Interpreter` will determine the configuration options.
  ///   - delegate: `Array` of `Delegate`s for the `Interpreter` to use to peform graph operations.
  ///       The default is `nil`.
  /// - Throws: An error if the model could not be loaded or the interpreter could not be created.
  public init(modelPath: String, options: Options? = nil, delegates: [Delegate]? = nil) throws {
    guard let model = Model(filePath: modelPath) else { throw InterpreterError.failedToLoadModel }
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
    guard let cInterpreter = TfLiteInterpreterCreate(model.cModel, cInterpreterOptions) else {
      throw InterpreterError.failedToCreateInterpreter
    }
    self.cInterpreter = cInterpreter
  }

  deinit {
    TfLiteInterpreterDelete(cInterpreter)
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
    guard TfLiteInterpreterResizeInputTensor(
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
}

extension Interpreter {
  /// Options for configuring the `Interpreter`.
  public struct Options: Equatable, Hashable {
    /// The maximum number of CPU threads that the interpreter should run on. The default is `nil`
    /// indicating that the `Interpreter` will decide the number of threads to use.
    public var threadCount: Int? = nil

    /// Creates a new instance with the default values.
    public init() {}
  }
}

/// A type alias for `Interpreter.Options` to support backwards compatiblity with the deprecated
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
    var buffer: UnsafeMutablePointer<CChar>?
    guard vasprintf(&buffer, cFormat, arguments) != 0, let cString = buffer else { return nil }
    self.init(validatingUTF8: cString)
  }
}
