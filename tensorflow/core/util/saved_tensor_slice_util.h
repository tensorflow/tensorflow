// Utilities for saving/restoring tensor slice checkpoints.

#ifndef TENSORFLOW_UTIL_SAVED_TENSOR_SLICE_UTIL_H_
#define TENSORFLOW_UTIL_SAVED_TENSOR_SLICE_UTIL_H_

#include <string>  // for string
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/public/status.h"  // for Status

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

template <typename T>
struct SaveTypeTraits;

template <typename T>
const typename SaveTypeTraits<T>::SavedType* TensorProtoData(
    const TensorProto& t);

template <typename T>
protobuf::RepeatedField<typename SaveTypeTraits<T>::SavedType>*
MutableTensorProtoData(TensorProto* t);

template <typename T>
void Fill(T* data, size_t n, TensorProto* t);

#define TENSOR_PROTO_EXTRACT_TYPE(TYPE, FIELD, FTYPE)                    \
  template <>                                                            \
  struct SaveTypeTraits<TYPE> {                                          \
    static constexpr bool supported = true;                              \
    typedef FTYPE SavedType;                                             \
  };                                                                     \
  template <>                                                            \
  inline const FTYPE* TensorProtoData<TYPE>(const TensorProto& t) {      \
    static_assert(SaveTypeTraits<TYPE>::supported,                       \
                  "Specified type " #TYPE " not supported for Restore"); \
    return reinterpret_cast<const FTYPE*>(t.FIELD##_val().data());       \
  }                                                                      \
  template <>                                                            \
  inline protobuf::RepeatedField<FTYPE>* MutableTensorProtoData<TYPE>(   \
      TensorProto * t) {                                                 \
    static_assert(SaveTypeTraits<TYPE>::supported,                       \
                  "Specified type " #TYPE " not supported for Save");    \
    return reinterpret_cast<protobuf::RepeatedField<FTYPE>*>(            \
        t->mutable_##FIELD##_val());                                     \
  }                                                                      \
  template <>                                                            \
  inline void Fill(const TYPE* data, size_t n, TensorProto* t) {         \
    typename protobuf::RepeatedField<FTYPE> copy(data, data + n);        \
    t->mutable_##FIELD##_val()->Swap(&copy);                             \
  }

TENSOR_PROTO_EXTRACT_TYPE(float, float, float);
TENSOR_PROTO_EXTRACT_TYPE(double, double, double);
TENSOR_PROTO_EXTRACT_TYPE(int32, int, int32);
TENSOR_PROTO_EXTRACT_TYPE(int64, int64, int64);
TENSOR_PROTO_EXTRACT_TYPE(uint8, int, int32);
TENSOR_PROTO_EXTRACT_TYPE(int8, int, int32);
TENSOR_PROTO_EXTRACT_TYPE(int16, int, int32);
TENSOR_PROTO_EXTRACT_TYPE(qint8, int, int32);
TENSOR_PROTO_EXTRACT_TYPE(quint8, int, int32);

#undef TENSOR_PROTO_EXTRACT_TYPE

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

}  // namespace checkpoint

}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_SAVED_TENSOR_SLICE_UTIL_H_
