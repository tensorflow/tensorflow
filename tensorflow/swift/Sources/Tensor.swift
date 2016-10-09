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

public protocol TensorDataUnit { }

extension Int8 : TensorDataUnit {}
extension Int16 : TensorDataUnit {}
extension Int32 : TensorDataUnit {}
extension Int64 : TensorDataUnit {}
extension Int : TensorDataUnit {}
extension Float : TensorDataUnit {}
extension Double : TensorDataUnit {}

public class Tensor<DataType: TensorDataUnit> {

    /// Type selection
    /// If we use safe type, namely generic tensor, we'll have
    /// to compare the dynamic type at run time.
    /// Consider replacing `TensorDataUnit` with a enum 
    /// and sacrifice type safety
    private var dataType: TF_DataType {
        if DataType.self == Int.self { return TF_INT64 }
        else if DataType.self == Float.self { return TF_FLOAT }
        else if DataType.self == Double.self { return TF_DOUBLE }
        else if DataType.self == Int8.self { return TF_INT8 }
        else if DataType.self == Int16.self { return TF_INT16 }
        else if DataType.self == Int32.self { return TF_INT32 }
        else if DataType.self == Int64.self { return TF_INT64 }
        else {
            fatalError("Unsupported type")
        }
    }

    init(shape: Shape, data: [DataType]) {
        var dims = shape.components.map{Int64($0)}
        var data = data
        TF_NewTensor(self.dataType, &dims, Int32(dims.count), &data, data.count, nil, nil)
    }
    
}
