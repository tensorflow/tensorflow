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

// Utilities for saving/restoring tensor slice checkpoints.

#ifndef TENSORFLOW_CORE_UTIL_SAVED_TENSOR_SLICE_UTIL_H_
#define TENSORFLOW_CORE_UTIL_SAVED_TENSOR_SLICE_UTIL_H_

#include <string>  // for string
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"  // for Status
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

namespace checkpoint {

// The key for the metadata in the tensor slice checkpoint files. It is "" so
// that the metadata is always at the beginning of a checkpoint file.
extern const char kSavedTensorSlicesKey[];

// Encode a tensor name + a tensor slice into an ordered code and outputs it as
// a string.
// The format is
//  <0>
//  <tensor_name>
//  <rank>
//  <dim-0-start><dim-0-length>
//  <dim-1-start><dim-1-length>
//  ...

string EncodeTensorNameSlice(const string& name,
                             const tensorflow::TensorSlice& slice);

// Parse out the name and the slice from string encoded as an ordered code.
Status DecodeTensorNameSlice(const string& code, string* name,
                             tensorflow::TensorSlice* slice);

// Extracts the full shape, slice spec, and shape of the slice from
// "shape_and_slice".  On non-OK return, caller must clear the out-arguments
// before reusing.
Status ParseShapeAndSlice(const string& shape_and_slice, TensorShape* shape,
                          TensorSlice* slice, TensorShape* shape_slice);

template <typename T>
struct SaveTypeTraits;

template <typename T>
const typename SaveTypeTraits<T>::SavedType* TensorProtoData(
    const TensorProto& t);

template <typename T>
typename SaveTypeTraits<T>::RepeatedField* MutableTensorProtoData(
    TensorProto* t);

template <typename T>
void Fill(T* data, size_t n, TensorProto* t);

#define TENSOR_PROTO_EXTRACT_TYPE_HELPER(TYPE, FIELD, FTYPE, STYPE)      \
  template <>                                                            \
  struct SaveTypeTraits<TYPE> {                                          \
    static constexpr bool supported = true;                              \
    typedef STYPE SavedType;                                             \
    typedef protobuf::RepeatedField<FTYPE> RepeatedField;                \
  };                                                                     \
  template <>                                                            \
  inline const STYPE* TensorProtoData<TYPE>(const TensorProto& t) {      \
    static_assert(SaveTypeTraits<TYPE>::supported,                       \
                  "Specified type " #TYPE " not supported for Restore"); \
    return reinterpret_cast<const STYPE*>(t.FIELD##_val().data());       \
  }                                                                      \
  template <>                                                            \
  inline protobuf::RepeatedField<FTYPE>* MutableTensorProtoData<TYPE>(   \
      TensorProto * t) {                                                 \
    static_assert(SaveTypeTraits<TYPE>::supported,                       \
                  "Specified type " #TYPE " not supported for Save");    \
    return reinterpret_cast<protobuf::RepeatedField<FTYPE>*>(            \
        t->mutable_##FIELD##_val());                                     \
  }

#define TENSOR_PROTO_EXTRACT_TYPE(TYPE, FIELD, FTYPE)             \
  TENSOR_PROTO_EXTRACT_TYPE_HELPER(TYPE, FIELD, FTYPE, FTYPE)     \
  template <>                                                     \
  inline void Fill(const TYPE* data, size_t n, TensorProto* t) {  \
    typename protobuf::RepeatedField<FTYPE> copy(data, data + n); \
    t->mutable_##FIELD##_val()->Swap(&copy);                      \
  }

// Complex needs special treatment since proto doesn't have native complex
#define TENSOR_PROTO_EXTRACT_TYPE_COMPLEX(TYPE, FIELD, FTYPE)       \
  TENSOR_PROTO_EXTRACT_TYPE_HELPER(TYPE, FIELD, FTYPE, TYPE)        \
  template <>                                                       \
  inline void Fill(const TYPE* data, size_t n, TensorProto* t) {    \
    const FTYPE* sub = reinterpret_cast<const FTYPE*>(data);        \
    typename protobuf::RepeatedField<FTYPE> copy(sub, sub + 2 * n); \
    t->mutable_##FIELD##_val()->Swap(&copy);                        \
  }

TENSOR_PROTO_EXTRACT_TYPE(bool, bool, bool);
TENSOR_PROTO_EXTRACT_TYPE(float, float, float);
TENSOR_PROTO_EXTRACT_TYPE(double, double, double);
TENSOR_PROTO_EXTRACT_TYPE_COMPLEX(complex64, scomplex, float);
TENSOR_PROTO_EXTRACT_TYPE_COMPLEX(complex128, dcomplex, double);
TENSOR_PROTO_EXTRACT_TYPE(int32, int, int32);
TENSOR_PROTO_EXTRACT_TYPE(int64, int64, protobuf_int64);
TENSOR_PROTO_EXTRACT_TYPE(uint16, int, int32);
TENSOR_PROTO_EXTRACT_TYPE(uint8, int, int32);
TENSOR_PROTO_EXTRACT_TYPE(int8, int, int32);
TENSOR_PROTO_EXTRACT_TYPE(int16, int, int32);
TENSOR_PROTO_EXTRACT_TYPE(qint8, int, int32);
TENSOR_PROTO_EXTRACT_TYPE(quint8, int, int32);
TENSOR_PROTO_EXTRACT_TYPE(quint16, int, int32);

#undef TENSOR_PROTO_EXTRACT_TYPE_COMPLEX
#undef TENSOR_PROTO_EXTRACT_TYPE_HELPER
#undef TENSOR_PROTO_EXTRACT_TYPE

// Custom implementation for qint32, based on the one for int32.

template <>
struct SaveTypeTraits<qint32> : SaveTypeTraits<int32> {};

template <>
inline const int32* TensorProtoData<qint32>(const TensorProto& t) {
  static_assert(SaveTypeTraits<qint32>::supported,
                "Specified type qint32 not supported for Restore");
  return reinterpret_cast<const int32*>(t.int_val().data());
}

inline void Fill(const qint32* data, size_t n, TensorProto* t) {
  const int32* p = reinterpret_cast<const int32*>(data);
  typename protobuf::RepeatedField<int32> copy(p, p + n);
  t->mutable_int_val()->Swap(&copy);
}

// Custom implementation for Eigen::half.

template <>
struct SaveTypeTraits<Eigen::half> {
  static constexpr bool supported = true;
  typedef int SavedType;
  typedef protobuf::RepeatedField<int32> RepeatedField;
};

template <>
inline const int* TensorProtoData<Eigen::half>(const TensorProto& t) {
  return t.half_val().data();
}

template <>
inline protobuf::RepeatedField<int32>* MutableTensorProtoData<Eigen::half>(
    TensorProto* t) {
  return t->mutable_half_val();
}

template <>
inline void Fill(const Eigen::half* data, size_t n, TensorProto* t) {
  typename protobuf::RepeatedField<int32>* val = t->mutable_half_val();
  val->Resize(n, 0);
  for (size_t i = 0; i < n; ++i) {
    val->Set(i, data[i].x);
  }
}

// Custom implementation for string.

template <>
struct SaveTypeTraits<tstring> {
  static constexpr bool supported = true;
  typedef const string* SavedType;
  typedef protobuf::RepeatedPtrField<string> RepeatedField;
};

template <>
inline const string* const* TensorProtoData<tstring>(const TensorProto& t) {
  static_assert(SaveTypeTraits<tstring>::supported,
                "Specified type tstring not supported for Restore");
  return t.string_val().data();
}

template <>
inline protobuf::RepeatedPtrField<string>* MutableTensorProtoData<tstring>(
    TensorProto* t) {
  static_assert(SaveTypeTraits<tstring>::supported,
                "Specified type tstring not supported for Save");
  return t->mutable_string_val();
}

template <>
inline void Fill(const tstring* data, size_t n, TensorProto* t) {
  typename protobuf::RepeatedPtrField<string> copy(data, data + n);
  t->mutable_string_val()->Swap(&copy);
}

}  // namespace checkpoint

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_SAVED_TENSOR_SLICE_UTIL_H_
