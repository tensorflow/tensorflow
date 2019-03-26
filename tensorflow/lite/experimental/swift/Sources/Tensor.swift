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
public struct Tensor {

  /// Name of the tensor.
  public let name: String

  /// Data type of the tensor.
  public let dataType: TensorDataType

  /// Shape of the tensor.
  public let shape: TensorShape

  /// Data in the input or output tensor.
  public let data: Data

  /// Quantization parameters for the tensor if using a quantized model.
  public let quantizationParameters: QuantizationParameters?

  /// Creates a new input or output tensor instance.
  ///
  /// - Parameters:
  ///   - name: Name of the tensor.
  ///   - dataType: Data type of the tensor.
  ///   - data: Data in the input tensor.
  ///   - quantizationParameters Quantization parameters for the tensor if using a quantized model.
  ///       The default is `nil`.
  init(
    name: String,
    dataType: TensorDataType,
    shape: TensorShape,
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

/// Supported TensorFlow Lite tensor data types.
public enum TensorDataType: Equatable {
  /// 32-bit single precision floating point tensor data type.
  case float32
  /// 8-bit unsigned integer tensor data type.
  case uInt8
  /// 16-bit signed integer tensor data type.
  case int16
  /// 32-bit signed integer tensor data type.
  case int32
  /// 64-bit signed integer tensor data type.
  case int64
  /// Boolean tensor data type.
  case bool

  /// Creates a new tensor data type from the given `TFL_Type` or `nil` if the data type is
  /// unsupported or could not be determined because there was an error.
  ///
  /// - Parameter type: A data type supported by a tensor.
  init?(type: TFL_Type) {
    switch type {
    case kTfLiteFloat32:
      self = .float32
    case kTfLiteUInt8:
      self = .uInt8
    case kTfLiteInt16:
      self = .int16
    case kTfLiteInt32:
      self = .int32
    case kTfLiteInt64:
      self = .int64
    case kTfLiteBool:
      self = .bool
    case kTfLiteNoType:
      fallthrough
    default:
      return nil
    }
  }
}

/// The shape of a TensorFlow Lite tensor.
public struct TensorShape {

  /// The number of dimensions of the tensor.
  public let rank: Int

  /// Array of dimensions for the tensor.
  public let dimensions: [Int]

  /// Array of `Int32` dimensions for the tensor.
  var int32Dimensions: [Int32] { return dimensions.map(Int32.init) }

  /// Creates a new tensor shape instance with the given array of dimensions.
  ///
  /// - Parameters:
  ///   - dimensions: Dimensions for the tensor.
  public init(_ dimensions: [Int]) {
    self.rank = dimensions.count
    self.dimensions = dimensions
  }

  /// Creates a new tensor shape instance with the given elements representing the dimensions.
  ///
  /// - Parameters:
  ///   - elements: Dimensions for the tensor.
  public init(_ elements: Int...) {
    self.init(elements)
  }
}

extension TensorShape: ExpressibleByArrayLiteral {
  /// Creates a new tensor shape instance with the given array literal representing the dimensions.
  ///
  /// - Parameters:
  ///   - arrayLiteral: Dimensions for the tensor.
  public init(arrayLiteral: Int...) {
    self.init(arrayLiteral)
  }
}
