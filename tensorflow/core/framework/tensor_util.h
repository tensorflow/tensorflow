/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_FRAMEWORK_TENSOR_UTIL_H_
#define TENSORFLOW_FRAMEWORK_TENSOR_UTIL_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

#include <vector>
namespace tensorflow {
namespace tensor {

// DeepCopy returns a tensor whose contents are a deep copy of the
// contents of 'other'.  This function is intended only for
// convenience, not speed.
//
// REQUIRES: 'other' must point to data stored in CPU memory.
// REQUIRES: 'other' must be a Tensor of a copy-able type if
//           'other' is not appropriately memory-aligned.
Tensor DeepCopy(const Tensor& other);

// Concatenates 'tensors' into a single tensor, along their 0th dimension.
//
// REQUIRES: All members of 'tensors' must have the same data type parameter.
// REQUIRES: Each member of 'tensors' must have at least one dimension.
// REQUIRES: Each member of 'tensors' must point to data stored in CPU memory.
// REQUIRES: Each member of 'tensors' must be a Tensor of a copy-able type if it
//           is not appropriately memory-aligned.
Status Concat(const gtl::ArraySlice<Tensor>& tensors,
              Tensor* result) TF_MUST_USE_RESULT;

// Splits 'tensor' into 'sizes.size()' individual tensors, along the 0th
// dimension. The ith output tensor has 0th-dimension size 'sizes[i]'.
//
// REQUIRES: 'tensor' must have at least one dimension.
// REQUIRES: 'tensor.dim_size(0)' must equal the sum of the elements of 'sizes'.
// REQUIRES: 'tensor' must point to data stored in CPU memory.
// REQUIRES: 'tensor' must be a Tensor of a copy-able type if it is not
//           appropriately memory-aligned.
//
// Split() and Concat() are inverse operations.
Status Split(const Tensor& tensor, const gtl::ArraySlice<int64>& sizes,
             std::vector<Tensor>* result) TF_MUST_USE_RESULT;

namespace internal {
void SetTensorProtoShape(std::vector<size_t> shape,
                         TensorShapeProto* shape_proto);

// Defines value type dependent methods to manipulate `TensorProto`.
// Class specializations has to define following methods:
//   static DataType GetDataType()
//   static void AddValue(Type value, TensorProto* proto)
template <typename Type>
class TensorProtoHelper : public std::false_type {};

template <>
class TensorProtoHelper<string> : public std::true_type {
 public:
  static DataType GetDataType() { return DataType::DT_STRING; }
  static void AddValue(const string& value, TensorProto* proto) {
    *proto->mutable_string_val()->Add() = value;
  }
};

template <>
class TensorProtoHelper<int32> : public std::true_type {
 public:
  static DataType GetDataType() { return DataType::DT_INT32; }
  static void AddValue(int32 value, TensorProto* proto) {
    proto->mutable_int_val()->Add(value);
  }
};

template <>
class TensorProtoHelper<int64> : public std::true_type {
 public:
  static DataType GetDataType() { return DataType::DT_INT64; }
  static void AddValue(int64 value, TensorProto* proto) {
    proto->mutable_int64_val()->Add(value);
  }
};

template <>
class TensorProtoHelper<uint32> : public std::true_type {
 public:
  static DataType GetDataType() { return DataType::DT_UINT32; }
  static void AddValue(uint32 value, TensorProto* proto) {
    proto->mutable_uint32_val()->Add(value);
  }
};

template <>
class TensorProtoHelper<uint64> : public std::true_type {
 public:
  static DataType GetDataType() { return DataType::DT_UINT64; }
  static void AddValue(uint64 value, TensorProto* proto) {
    proto->mutable_uint64_val()->Add(value);
  }
};

template <>
class TensorProtoHelper<float> : public std::true_type {
 public:
  static DataType GetDataType() { return DataType::DT_FLOAT; }
  static void AddValue(float value, TensorProto* proto) {
    proto->mutable_float_val()->Add(value);
  }
};

template <>
class TensorProtoHelper<double> : public std::true_type {
 public:
  static DataType GetDataType() { return DataType::DT_DOUBLE; }
  static void AddValue(double value, TensorProto* proto) {
    proto->mutable_double_val()->Add(value);
  }
};

template <>
class TensorProtoHelper<bool> : public std::true_type {
 public:
  static DataType GetDataType() { return DataType::DT_BOOL; }
  static void AddValue(bool value, TensorProto* proto) {
    proto->mutable_bool_val()->Add(value);
  }
};
}  // namespace internal

// Creates a 'TensorProto' with specified shape and values.
// The dtype and a field to represent data values of the returned 'TensorProto'
// are determined based on type of the 'values' parameter.
template <typename Type>
typename std::enable_if<internal::TensorProtoHelper<Type>::value,
                        TensorProto>::type
CreateTensorProto(const std::vector<Type>& values,
                  const std::vector<size_t>& shape) {
  TensorProto tensor;
  using TypeHelper = internal::TensorProtoHelper<Type>;
  tensor.set_dtype(TypeHelper::GetDataType());
  internal::SetTensorProtoShape(shape, tensor.mutable_tensor_shape());
  for (const auto& value : values) {
    TypeHelper::AddValue(value, &tensor);
  }
  return tensor;
}

}  // namespace tensor
}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_TENSOR_UTIL_H_
