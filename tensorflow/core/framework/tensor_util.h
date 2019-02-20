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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_UTIL_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_UTIL_H_

#include <algorithm>
#include <vector>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

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
// Class specializations have to define following methods:
//   static DataType GetDataType()
//   static void AddValue(Type value, TensorProto* proto)
//   template <typename IterType>
//   static void AddValues(IterType begin, IterType end, TensorProto* proto)

template <typename Type>
class TensorProtoHelper : public std::false_type {};

#define DEFINE_PROTO_HELPER(TYPE, TF_TYPE, FIELDTYPE)                         \
  template <>                                                                 \
  class TensorProtoHelper<TYPE> : public std::true_type {                     \
   public:                                                                    \
    static DataType GetDataType() { return DataType::TF_TYPE; }               \
    static void AddValue(const TYPE& value, TensorProto* proto) {             \
      proto->mutable_##FIELDTYPE##_val()->Add(value);                         \
    }                                                                         \
    template <typename IterType>                                              \
    static void AddValues(IterType begin, IterType end, TensorProto* proto) { \
      using SrcType = typename std::iterator_traits<IterType>::value_type;    \
      size_t n = std::distance(begin, end);                                   \
      FIELDTYPE* dst_ptr = AppendUninitialized(n, proto);                     \
      if (std::is_same<SrcType, FIELDTYPE>::value) {                          \
        std::copy(begin, end, dst_ptr);                                       \
      } else {                                                                \
        std::transform(begin, end, dst_ptr, [](SrcType x) -> FIELDTYPE {      \
          return static_cast<FIELDTYPE>(x);                                   \
        });                                                                   \
      }                                                                       \
    }                                                                         \
                                                                              \
   private:                                                                   \
    static FIELDTYPE* AppendUninitialized(size_t n, TensorProto* proto) {     \
      auto* field = proto->mutable_##FIELDTYPE##_val();                       \
      field->Reserve(field->size() + n);                                      \
      return reinterpret_cast<FIELDTYPE*>(field->AddNAlreadyReserved(n));     \
    }                                                                         \
  }

DEFINE_PROTO_HELPER(float, DT_FLOAT, float);
DEFINE_PROTO_HELPER(double, DT_DOUBLE, double);
DEFINE_PROTO_HELPER(int8, DT_INT8, int);
DEFINE_PROTO_HELPER(uint8, DT_UINT8, int);
DEFINE_PROTO_HELPER(int16, DT_INT16, int);
DEFINE_PROTO_HELPER(uint16, DT_UINT16, int);
DEFINE_PROTO_HELPER(int32, DT_INT32, int);
DEFINE_PROTO_HELPER(uint32, DT_UINT32, uint32);
DEFINE_PROTO_HELPER(int64, DT_INT64, int64);
DEFINE_PROTO_HELPER(uint64, DT_UINT64, uint64);
DEFINE_PROTO_HELPER(bool, DT_BOOL, bool);

#undef DEFINE_PROTO_HELPER

template <>
class TensorProtoHelper<string> : public std::true_type {
 public:
  static DataType GetDataType() { return DataType::DT_STRING; }
  static void AddValue(const string& value, TensorProto* proto) {
    *proto->mutable_string_val()->Add() = value;
  }
  template <typename IterType>
  static void AddValues(IterType begin, IterType end, TensorProto* proto) {
    for (IterType it = begin; it != end; ++it) {
      AddValue(*it, proto);
    }
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
  TypeHelper::AddValues(values.begin(), values.end(), &tensor);
  return tensor;
}

}  // namespace tensor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_UTIL_H_
