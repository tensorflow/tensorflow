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

/// An input or output tensor in a TensorFlow Lite graph.
public struct Tensor: Equatable, Hashable {
  /// The name of the `Tensor`.
  public let name: String

  /// The data type of the `Tensor`.
  public let dataType: DataType

  /// The shape of the `Tensor`.
  public let shape: Shape

  /// The data in the input or output `Tensor`.
  public let data: Data

  /// The quantization parameters for the `Tensor` if using a quantized model.
  public let quantizationParameters: QuantizationParameters?

  /// Creates a new input or output `Tensor` instance.
  ///
  /// - Parameters:
  ///   - name: The name of the `Tensor`.
  ///   - dataType: The data type of the `Tensor`.
  ///   - shape: The shape of the `Tensor`.
  ///   - data: The data in the input `Tensor`.
  ///   - quantizationParameters Parameters for the `Tensor` if using a quantized model. The default
  ///       is `nil`.
  init(
    name: String,
    dataType: DataType,
    shape: Shape,
    data: Data,
    quantizationParameters: QuantizationParameters? = nil
  ) {
    self.name = name
    self.dataType = dataType
    self.shape = shape
    self.data = data
    self.quantizationParameters = quantizationParameters
  }
}

extension Tensor {
  /// The supported `Tensor` data types.
  public enum DataType: Equatable, Hashable {
    /// A boolean.
    case bool
    /// An 8-bit unsigned integer.
    case uInt8
    /// A 16-bit signed integer.
    case int16
    /// A 32-bit signed integer.
    case int32
    /// A 64-bit signed integer.
    case int64
    /// A 16-bit half precision floating point.
    case float16
    /// A 32-bit single precision floating point.
    case float32
    /// A 64-bit double precision floating point.
    case float64

    /// Creates a new instance from the given `TfLiteType` or `nil` if the data type is unsupported
    /// or could not be determined because there was an error.
    ///
    /// - Parameter type: A data type for a tensor.
    init?(type: TfLiteType) {
      switch type {
      case kTfLiteBool:
        self = .bool
      case kTfLiteUInt8:
        self = .uInt8
      case kTfLiteInt16:
        self = .int16
      case kTfLiteInt32:
        self = .int32
      case kTfLiteInt64:
        self = .int64
      case kTfLiteFloat16:
        self = .float16
      case kTfLiteFloat32:
        self = .float32
      case kTfLiteFloat64:
        self = .float64
      case kTfLiteNoType:
        fallthrough
      default:
        return nil
      }
    }
  }
}

extension Tensor {
  /// The shape of a `Tensor`.
  public struct Shape: Equatable, Hashable {
    /// The number of dimensions of the `Tensor`.
    public let rank: Int

    /// An array of dimensions for the `Tensor`.
    public let dimensions: [Int]

    /// An array of `Int32` dimensions for the `Tensor`.
    var int32Dimensions: [Int32] { return dimensions.map(Int32.init) }

    /// Creates a new instance with the given array of dimensions.
    ///
    /// - Parameters:
    ///   - dimensions: Dimensions for the `Tensor`.
    public init(_ dimensions: [Int]) {
      self.rank = dimensions.count
      self.dimensions = dimensions
    }

    /// Creates a new instance with the given elements representing the dimensions.
    ///
    /// - Parameters:
    ///   - elements: Dimensions for the `Tensor`.
    public init(_ elements: Int...) {
      self.init(elements)
    }
  }
}

extension Tensor.Shape: ExpressibleByArrayLiteral {
  /// Creates a new instance with the given array literal representing the dimensions.
  ///
  /// - Parameters:
  ///   - arrayLiteral: Dimensions for the `Tensor`.
  public init(arrayLiteral: Int...) {
    self.init(arrayLiteral)
  }
}
