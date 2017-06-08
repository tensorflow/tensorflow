/// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
///
/// Licensed under the Apache License, Version 2.0 (the "License");
/// you may not use this file except in compliance with the License.
/// You may obtain a copy of the License at
///
/// http://www.apache.org/licenses/LICENSE-2.0
///
/// Unless required by applicable law or agreed to in writing, software
/// distributed under the License is distributed on an "AS IS" BASIS,
/// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
/// See the License for the specific language governing permissions and
/// limitations under the License.
/// ==============================================================================

import CTensorFlow

public enum TensorDataType : UInt32 {
    case float = 1
    case double = 2
    case int32 = 3
    case int16 = 5
    case int8 = 6
    case int64 = 9
}

public protocol TensorDataProtocol {
    static var dataType: TensorDataType { get }
}

extension TensorDataProtocol {
    static var cType: TF_DataType {
        return TF_DataType(Self.dataType.rawValue)
    }
}

extension Int8 : TensorDataProtocol {
    public static var dataType: TensorDataType {
        return .int8
    }
}

extension Int16 : TensorDataProtocol {
    public static var dataType: TensorDataType {
        return .int16
    }

}

extension Int32 : TensorDataProtocol {
    public static var dataType: TensorDataType {
        return .int32
    }
}

extension Int64 : TensorDataProtocol {
    public static var dataType: TensorDataType {
        return .int64
    }
}

extension Int : TensorDataProtocol {
    public static var dataType: TensorDataType {
        return .int64
    }
}

extension Float : TensorDataProtocol {
    public static var dataType: TensorDataType {
        return .float
    }
}

extension Double : TensorDataProtocol {

    public static var dataType: TensorDataType {
        return .double
    }
}

public class Tensor<Element: TensorDataProtocol> {

    init(shape: Shape, data: [Element]) {
        var dims = shape.components.map{Int64($0)}
        var data = data
        TF_NewTensor(Element.cType, &dims, Int32(dims.count), &data, data.count, nil, nil)
    }

}
