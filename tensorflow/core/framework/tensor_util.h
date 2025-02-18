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
#include "tensorflow/core/framework/type_traits.h"
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

// Deep copies input to output.  This function is similar to above, but assumes
// that the memory for the output has already been allocated.
void DeepCopy(const Tensor& input, Tensor* output);

// Concatenates 'tensors' into a single tensor, along their 0th dimension.
//
// REQUIRES: All members of 'tensors' must have the same data type parameter.
// REQUIRES: Each member of 'tensors' must have at least one dimension.
// REQUIRES: Each member of 'tensors' must point to data stored in CPU memory.
// REQUIRES: Each member of 'tensors' must be a Tensor of a copy-able type if it
//           is not appropriately memory-aligned.
absl::Status Concat(absl::Span<const Tensor> tensors, Tensor* result);

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
absl::Status Split(const Tensor& tensor, absl::Span<const int64_t> sizes,
                   std::vector<Tensor>* result);

namespace internal {
void SetTensorProtoShape(absl::Span<const size_t> shape,
                         TensorShapeProto* shape_proto);

template <typename Type>
class TensorProtoFieldHelper : public std::false_type {};

#define DEFINE_PROTO_FIELD_HELPER(TYPE, FIELDNAME)                            \
  template <>                                                                 \
  class TensorProtoFieldHelper<TYPE> : public std::true_type {                \
   public:                                                                    \
    typedef decltype(                                                         \
        std::declval<TensorProto>().FIELDNAME##_val(0)) FieldType;            \
    typedef decltype(                                                         \
        std::declval<TensorProto>().FIELDNAME##_val()) RepeatedFieldType;     \
    typedef decltype(std::declval<TensorProto>().mutable_##FIELDNAME##_val()) \
        MutableRepeatedFieldType;                                             \
    static MutableRepeatedFieldType GetMutableField(TensorProto* proto) {     \
      return proto->mutable_##FIELDNAME##_val();                              \
    }                                                                         \
    static RepeatedFieldType& GetField(const TensorProto& proto) {            \
      return proto.FIELDNAME##_val();                                         \
    }                                                                         \
  }

// The argument pairs in the following macro instantiations encode the
// mapping from C++ type ($1) to repeated field name "$2_val" used for storing
// values in TensorProto. See tensorflow/core/framework/tensor.proto.
DEFINE_PROTO_FIELD_HELPER(float, float);
DEFINE_PROTO_FIELD_HELPER(double, double);
DEFINE_PROTO_FIELD_HELPER(int8, int);
DEFINE_PROTO_FIELD_HELPER(uint8, int);
DEFINE_PROTO_FIELD_HELPER(int16, int);
DEFINE_PROTO_FIELD_HELPER(uint16, int);
DEFINE_PROTO_FIELD_HELPER(int32, int);
DEFINE_PROTO_FIELD_HELPER(uint32, uint32);
DEFINE_PROTO_FIELD_HELPER(int64_t, int64);
DEFINE_PROTO_FIELD_HELPER(uint64, uint64);
DEFINE_PROTO_FIELD_HELPER(bool, bool);
DEFINE_PROTO_FIELD_HELPER(qint8, int);
DEFINE_PROTO_FIELD_HELPER(quint8, int);
DEFINE_PROTO_FIELD_HELPER(qint16, int);
DEFINE_PROTO_FIELD_HELPER(quint16, int);
DEFINE_PROTO_FIELD_HELPER(qint32, int);
DEFINE_PROTO_FIELD_HELPER(Eigen::half, half);
DEFINE_PROTO_FIELD_HELPER(bfloat16, half);
DEFINE_PROTO_FIELD_HELPER(complex64, scomplex);
DEFINE_PROTO_FIELD_HELPER(complex128, dcomplex);

#undef DEFINE_PROTO_HELPER

template <typename T>
struct CopyHelper {
  template <typename SrcIter, typename DstIter>
  static void ToArray(SrcIter begin, SrcIter end, DstIter dst) {
    using SrcType = typename std::iterator_traits<SrcIter>::value_type;
    using DstType = typename std::iterator_traits<DstIter>::value_type;
    std::transform(begin, end, dst, [](const SrcType& x) -> DstType {
      return static_cast<DstType>(x);
    });
  }
  template <typename SrcIter>
  static void ToArray(SrcIter begin, SrcIter end, SrcIter dst) {
    std::copy(begin, end, dst);
  }
  template <typename SrcIter, typename DstIter>
  static void FromArray(SrcIter begin, SrcIter end, DstIter dst) {
    ToArray(begin, end, dst);
  }
};

// Overloads for Eigen::half and bfloat16 that are 16 bits in size but are
// stored in an int32 field.
template <>
struct CopyHelper<Eigen::half> {
  template <typename SrcIter>
  static void ToArray(SrcIter begin, SrcIter end, Eigen::half* dst) {
    std::transform(begin, end, dst, [](int x) -> Eigen::half {
      return Eigen::numext::bit_cast<Eigen::half>(static_cast<uint16>(x));
    });
  }
  template <typename SrcIter, typename DstIter>
  static void FromArray(SrcIter begin, SrcIter end, DstIter dst) {
    std::transform(begin, end, dst, [](Eigen::half h) -> int {
      return static_cast<int>(Eigen::numext::bit_cast<uint16>(h));
    });
  }
};

template <>
struct CopyHelper<bfloat16> {
  template <typename SrcIter>
  static void ToArray(SrcIter begin, SrcIter end, bfloat16* dst) {
    std::transform(begin, end, dst, [](int x) -> bfloat16 {
      return Eigen::numext::bit_cast<bfloat16>(static_cast<uint16>(x));
    });
  }
  template <typename SrcIter, typename DstIter>
  static void FromArray(SrcIter begin, SrcIter end, DstIter dst) {
    std::transform(begin, end, dst, [](bfloat16 bf16) -> int {
      return static_cast<int>(Eigen::numext::bit_cast<uint16>(bf16));
    });
  }
};

// Overloads for complex types that store real and imaginary parts
// at indices 2*i and 2*i+1 in float or double field.
template <typename RealType>
struct CopyHelper<std::complex<RealType>> {
  template <typename SrcIter>
  static void ToArray(SrcIter begin, SrcIter end, std::complex<RealType>* dst) {
    RealType* real_dst = reinterpret_cast<RealType*>(dst);
    std::copy(begin, end, real_dst);
  }

  template <typename SrcIter, typename DstIter>
  static void FromArray(SrcIter begin, SrcIter end, DstIter dst) {
    size_t n = std::distance(begin, end);
    const RealType* real_begin = reinterpret_cast<const RealType*>(&(*begin));
    std::copy_n(real_begin, 2 * n, dst);
  }
};

// Helper class to extract and insert values into TensorProto represented as
// repeated fields.
template <typename T>
class TensorProtoHelper : public std::true_type {
 public:
  using FieldHelper = TensorProtoFieldHelper<T>;
  using FieldType = typename TensorProtoFieldHelper<T>::FieldType;

  static DataType GetDataType() { return DataTypeToEnum<T>::value; }

  // Returns the number of values of type T encoded in the proto.
  static size_t NumValues(const TensorProto& proto) {
    size_t raw_size = FieldHelper::GetField(proto).size();
    return is_complex<T>::value ? raw_size / 2 : raw_size;
  }

  static void AddValue(const T& value, TensorProto* proto) {
    const T* val_ptr = &value;
    AddValues(val_ptr, val_ptr + 1, proto);
  }

  static T GetValue(size_t index, const TensorProto& proto) {
    const size_t stride = is_complex<T>::value ? 2 : 1;
    T val;
    CopyHelper<T>::ToArray(
        FieldHelper::GetField(proto).begin() + stride * index,
        FieldHelper::GetField(proto).begin() + stride * (index + 1), &val);
    return val;
  }

  template <typename IterType>
  static void AddValues(IterType begin, IterType end, TensorProto* proto) {
    size_t n = std::distance(begin, end);
    FieldType* dst = AppendUninitialized(n, proto);
    CopyHelper<T>::FromArray(begin, end, dst);
  }

  template <typename IterType>
  static void CopyValues(IterType dst, const TensorProto& proto) {
    CopyHelper<T>::ToArray(FieldHelper::GetField(proto).begin(),
                           FieldHelper::GetField(proto).end(), dst);
  }

  static void Truncate(size_t new_size, TensorProto* proto) {
    if (is_complex<T>::value) new_size *= 2;
    FieldHelper::GetMutableField(proto)->Truncate(new_size);
  }

  static FieldType* AppendUninitialized(size_t n, TensorProto* proto) {
    if (is_complex<T>::value) n *= 2;
    auto* field = FieldHelper::GetMutableField(proto);
    field->Reserve(field->size() + n);
    return reinterpret_cast<FieldType*>(field->AddNAlreadyReserved(n));
  }
};

// Specialization for string.
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
  template <typename IterType>
  static void CopyToTensorContent(IterType begin, IterType end,
                                  TensorProto* proto) {
    AddValues(begin, end, proto);
  }
};

template <typename Type, typename IterType>
typename std::enable_if<internal::TensorProtoHelper<Type>::value,
                        TensorProto>::type
CreateTensorProto(IterType values_begin, IterType values_end,
                  const size_t values_size,
                  const absl::Span<const size_t> shape) {
  TensorProto tensor;
  TensorShapeProto tensor_shape_proto;
  internal::SetTensorProtoShape(shape, &tensor_shape_proto);
  if (TensorShape(tensor_shape_proto).num_elements() != values_size) {
    LOG(ERROR) << "Shape and number of values (" << values_size
               << ") are incompatible.";
    return tensor;
  }
  using TypeHelper = internal::TensorProtoHelper<Type>;
  tensor.set_dtype(TypeHelper::GetDataType());
  *tensor.mutable_tensor_shape() = std::move(tensor_shape_proto);
  TypeHelper::AddValues(values_begin, values_end, &tensor);
  return tensor;
}

}  // namespace internal

// Creates a 'TensorProto' with the specified shape and values. The dtype and a
// field to represent data values of the returned 'TensorProto' are determined
// based on Type. Note that unless the argument provided to `values` is already
// an absl::Span, `Type` will need to be provided as a template parameter--the
// compiler can't infer it:
//   auto proto = CreateTensorProtoSpan<float>(my_array, shape);
template <typename Type>
typename std::enable_if<internal::TensorProtoHelper<Type>::value,
                        TensorProto>::type
CreateTensorProtoSpan(const absl::Span<const Type> values,
                      const absl::Span<const size_t> shape) {
  return internal::CreateTensorProto<Type>(values.begin(), values.end(),
                                           values.size(), shape);
}

// Version of the above that's more convenient if `values` is an std::vector, in
// which case Type can automatically be inferred:
//   auto proto = CreateTensorProto(my_vector, shape);
template <typename Type>
typename std::enable_if<internal::TensorProtoHelper<Type>::value,
                        TensorProto>::type
CreateTensorProto(const std::vector<Type>& values,
                  const absl::Span<const size_t> shape) {
  // This awkward iterator passing is essentially just to support vector<bool>,
  // otherwise we could just represent the vector as a Span.
  return internal::CreateTensorProto<Type>(values.begin(), values.end(),
                                           values.size(), shape);
}

// Converts values in tensor to run-length encoded compressed form.
//
// The elements of a tensor can be stored in a TensorProto in one of the
// following two forms:
// 1. As a raw byte string in the field `tensor_content` containing the
//    serialized in-memory representation of the tensor.
// 2. As values of a repeated field depending on the datatype, e.g. that
//    values of a DT_FLOAT tensor would be stored in the repeated field
//    `float_val`.
// Storage scheme 2 may use a simple form of run-length encoding to compress
// data: If the values contains a tail of identical values, the repeated field
// will be truncated such that the number of values in the repeated field is
// less than the number of elements implied by the field`tensor_shape`. The
// original tensor can be recovered by repeating the final value in the repeated
// field.
//
// The TensorProto will be compressed if a) the tensor contains at least
// min_num_elements elements and b) the compressed tensor proto is would be at
// most the size of the original tensor proto divided by min_compression_ratio.
//
// Returns true if the tensor was compressed.
bool CompressTensorProtoInPlace(int64_t min_num_elements,
                                float min_compression_ratio,
                                TensorProto* tensor);

inline bool CompressTensorProtoInPlace(TensorProto* tensor) {
  static const int64_t kDefaultMinNumElements = 64;
  static const float kDefaultMinCompressionRatio = 2.0f;
  return CompressTensorProtoInPlace(kDefaultMinNumElements,
                                    kDefaultMinCompressionRatio, tensor);
}

// Make a TensorShape from the contents of shape_t. Shape_t must be a
// 1-dimensional tensor of type int32 or int64.
absl::Status MakeShape(const Tensor& shape_t, TensorShape* out);

}  // namespace tensor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_UTIL_H_
