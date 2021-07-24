/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/batch_util.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

#define TF_CALL_DATASET_TYPES(m) TF_CALL_ALL_TYPES(m) TF_CALL_QUANTIZED_TYPES(m)

namespace tensorflow {
namespace batch_util {

namespace {

Status ValidateInput(const Tensor& parent, const Tensor& element,
                     int64_t index) {
  DCHECK_NE(parent.dim_size(0), 0);
  DCHECK_GE(index, 0);
  if (element.NumElements() != (parent.NumElements() / parent.dim_size(0))) {
    TensorShape chip_shape = parent.shape();
    chip_shape.RemoveDim(0);
    return errors::Internal(
        "ValidateInput Cannot perform copy: number of elements does not match. "
        " Shapes are: [element]: ",
        element.shape().DebugString(),
        ", [parent slice]: ", chip_shape.DebugString());
  }
  return Status::OK();
}

template <typename T>
Status HandleElementToSlice(const Tensor& /* element */, T* src, T* dest,
                            int64_t num_values) {
  static_assert(is_simple_type<T>::value, "Memcpy requires a simple type.");
  memcpy(dest, src, num_values * sizeof(T));
  return Status::OK();
}

template <>
Status HandleElementToSlice<tstring>(const Tensor& element, tstring* src,
                                     tstring* dest, int64_t num_values) {
  if (element.RefCountIsOne()) {
    for (int64_t i = 0; i < num_values; ++i) {
      *dest++ = std::move(*src++);
    }
  } else {
    std::copy_n(src, num_values, dest);
  }
  return Status::OK();
}

template <>
Status HandleElementToSlice<Variant>(const Tensor& element, Variant* src,
                                     Variant* dest, int64_t num_values) {
  if (element.RefCountIsOne()) {
    for (int64_t i = 0; i < num_values; ++i) {
      *dest++ = std::move(*src++);
    }
  } else {
    std::copy_n(src, num_values, dest);
  }
  return Status::OK();
}

template <>
Status HandleElementToSlice<ResourceHandle>(const Tensor& /* element */,
                                            ResourceHandle* src,
                                            ResourceHandle* dest,
                                            int64_t num_values) {
  std::copy_n(src, num_values, dest);
  return Status::OK();
}

template <>
Status HandleElementToSlice<Eigen::half>(const Tensor& /* element */,
                                         Eigen::half* src, Eigen::half* dest,
                                         int64_t num_values) {
  std::copy_n(src, num_values, dest);
  return Status::OK();
}

template <typename T>
void HandleSliceToElement(const T* src, T* dest, int64_t num_values) {
  static_assert(is_simple_type<T>::value, "Memcpy requires a simple type.");
  memcpy(dest, src, num_values * sizeof(T));
}

template <>
void HandleSliceToElement<tstring>(const tstring* src, tstring* dest,
                                   int64_t num_values) {
  std::copy_n(src, num_values, dest);
}

template <>
void HandleSliceToElement<Variant>(const Variant* src, Variant* dest,
                                   int64_t num_values) {
  std::copy_n(src, num_values, dest);
}

template <>
void HandleSliceToElement<ResourceHandle>(const ResourceHandle* src,
                                          ResourceHandle* dest,
                                          int64_t num_values) {
  std::copy_n(src, num_values, dest);
}

template <>
void HandleSliceToElement<Eigen::half>(const Eigen::half* src,
                                       Eigen::half* dest, int64_t num_values) {
  std::copy_n(src, num_values, dest);
}

template <typename T>
void HandleSliceToElement(Tensor* parent, T* src, T* dest, int64_t num_values) {
  static_assert(is_simple_type<T>::value, "Memcpy requires a simple type.");
  memcpy(dest, src, num_values * sizeof(T));
}

template <>
void HandleSliceToElement<tstring>(Tensor* parent, tstring* src, tstring* dest,
                                   int64_t num_values) {
  if (parent->RefCountIsOne()) {
    for (int64_t i = 0; i < num_values; ++i) {
      dest[i] = std::move(src[i]);
    }
  } else {
    std::copy_n(src, num_values, dest);
  }
}

template <>
void HandleSliceToElement<Variant>(Tensor* parent, Variant* src, Variant* dest,
                                   int64_t num_values) {
  if (parent->RefCountIsOne()) {
    for (int64_t i = 0; i < num_values; ++i) {
      dest[i] = std::move(src[i]);
    }
  } else {
    std::copy_n(src, num_values, dest);
  }
}

template <>
void HandleSliceToElement<ResourceHandle>(Tensor* parent, ResourceHandle* src,
                                          ResourceHandle* dest,
                                          int64_t num_values) {
  std::copy_n(src, num_values, dest);
}

template <>
void HandleSliceToElement<Eigen::half>(Tensor* parent, Eigen::half* src,
                                       Eigen::half* dest, int64_t num_values) {
  std::copy_n(src, num_values, dest);
}

}  // namespace

// Copies element into the index^th slice of parent (in the 0th dimension).
Status CopyElementToSlice(Tensor element, Tensor* parent, int64_t index) {
  TF_RETURN_IF_ERROR(ValidateInput(*parent, element, index));
  const int64_t num_values = element.NumElements();
#define HANDLE_TYPE(T)                                              \
  case DataTypeToEnum<T>::value: {                                  \
    T* src = element.base<T>();                                     \
    T* dest = parent->base<T>() + (num_values * index);             \
    return HandleElementToSlice<T>(element, src, dest, num_values); \
  }

  switch (element.dtype()) {
    TF_CALL_ALL_TYPES(HANDLE_TYPE);
    TF_CALL_QUANTIZED_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
    default:
      return errors::Unimplemented("CopyElementToSlice Unhandled data type: ",
                                   element.dtype());
  }
}

// Copies the index^th slice of parent (in the 0th dimension) into element.
Status CopySliceToElement(const Tensor& parent, Tensor* element,
                          int64_t index) {
  TF_RETURN_IF_ERROR(ValidateInput(parent, *element, index));
  const int64_t num_values = element->NumElements();

#define HANDLE_TYPE(T)                                      \
  case DataTypeToEnum<T>::value: {                          \
    const T* src = parent.base<T>() + (num_values * index); \
    T* dest = element->base<T>();                           \
    HandleSliceToElement<T>(src, dest, num_values);         \
    return Status::OK();                                    \
  }

  switch (parent.dtype()) {
    TF_CALL_ALL_TYPES(HANDLE_TYPE);
    TF_CALL_QUANTIZED_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
    default:
      return errors::Unimplemented("CopySliceToElement Unhandled data type: ",
                                   element->dtype());
  }
}

Status CopyContiguousSlices(const Tensor& src, int64_t src_offset,
                            int64_t dst_offset, int64_t num_slices,
                            Tensor* dst) {
  if (src.dtype() != dst->dtype()) {
    return errors::FailedPrecondition(
        "CopyContiguousSlices cannot perform copy: src and dst have different "
        "dtypes. Source dtype: ",
        src.dtype(), " dstination dtype: ", dst->dtype(), ".");
  }
  if (src.dims() < 1) {
    return errors::FailedPrecondition(
        "CopyContiguousSlices cannot perform copy: src has to be a tensor with "
        "rank >= 1. Source shape: ",
        src.shape().DebugString());
  }

  if (dst->dims() < 1) {
    return errors::FailedPrecondition(
        "CopyContiguousSlices cannot perform copy: dst has to be a tensor "
        "with rank >= 1. Dest shape: ",
        dst->shape().DebugString());
  }

  const int64_t src_dim0 = src.dim_size(0);
  const int64_t dst_dim0 = dst->dim_size(0);
  int64_t src_chip_size = 1;
  int64_t dst_chip_size = 1;
  for (int i = 1; i < src.dims(); ++i) {
    src_chip_size *= src.dim_size(i);
  }
  for (int i = 1; i < dst->dims(); ++i) {
    dst_chip_size *= dst->dim_size(i);
  }

  if (src_chip_size != dst_chip_size) {
    return errors::FailedPrecondition(
        "CopyContiguousSlices cannot perform copy: source and dst shapes are"
        "not compatible. Source shape: ",
        src.shape().DebugString(), ", dst shape: ", dst->shape().DebugString());
  }

  if (src_chip_size == 0 && dst_chip_size == 0) {
    return Status::OK();
  }

  if (src_offset < 0 || src_offset + num_slices > src_dim0 || dst_offset < 0 ||
      dst_offset + num_slices > dst_dim0) {
    return errors::FailedPrecondition(
        "CopyContiguousSlices cannot perform copy: index out of range. "
        "src_offset: ",
        src_offset, ", num_slices: ", num_slices, ", src_dim0: ", src_dim0,
        ", dst_offset: ", dst_offset, ", dst_dim0: ", dst_dim0, ".");
  }

#define HANDLE_TYPE(T)                                                 \
  case DataTypeToEnum<T>::value: {                                     \
    const T* src_p = src.base<T>() + (src_chip_size * src_offset);     \
    T* dst_p = dst->base<T>() + (dst_chip_size * dst_offset);          \
    HandleSliceToElement<T>(src_p, dst_p, src_chip_size * num_slices); \
    return Status::OK();                                               \
  }

  switch (src.dtype()) {
    TF_CALL_ALL_TYPES(HANDLE_TYPE);
    TF_CALL_QUANTIZED_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
    default:
      return errors::Unimplemented("CopyContiguousSlices unhandled data type: ",
                                   src.dtype());
  }
}

// Copies the index^th slice of parent (in the 0th dimension) into element.
//
// NOTE(mrry): The implementation may be able to optimize the copy to a move.
// This is particularly important for DT_STRING tensors.
Status MaybeMoveSliceToElement(Tensor* parent, Tensor* element, int64_t index) {
  TF_RETURN_IF_ERROR(ValidateInput(*parent, *element, index));
  const int64_t num_values = element->NumElements();

#define HANDLE_TYPE(T)                                      \
  case DataTypeToEnum<T>::value: {                          \
    T* src = parent->base<T>() + (num_values * index);      \
    T* dest = element->base<T>();                           \
    HandleSliceToElement<T>(parent, src, dest, num_values); \
    return Status::OK();                                    \
  }

  switch (parent->dtype()) {
    TF_CALL_ALL_TYPES(HANDLE_TYPE);
    TF_CALL_QUANTIZED_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
    default:
      return errors::Unimplemented(
          "MaybeMoveSliceToElement Unhandled data type: ", element->dtype());
  }
}

// The following five functions are copied from padding_fifo_queue.cc.
// TODO(mrry): Reconcile these functions with the similar methods in the
// queue implementation.
Status ValidateElementToLargerSlice(const Tensor& element, Tensor* parent) {
  DCHECK_NE(parent->dim_size(0), 0);
  if (element.NumElements() > (parent->NumElements() / parent->dim_size(0))) {
    TensorShape chip_shape = parent->shape();
    chip_shape.RemoveDim(0);
    return errors::Internal(
        "HandleElementToLargerSlice Cannot copy slice: number of entries in "
        "element is greater than number of elements in parent slice.  ",
        "Shapes are: [element]: ", element.shape().DebugString(),
        ", [parent slice]: ", chip_shape.DebugString());
  }
  return Status::OK();
}

template <typename T, int NDIMS>
Status HandleElementToLargerSlice(const Tensor& element, Tensor* parent,
                                  int index) {
  TF_RETURN_IF_ERROR(ValidateElementToLargerSlice(element, parent));
  if (element.NumElements() == 0) {
    return Status::OK();
  }
  auto element_t = element.tensor<T, NDIMS>();
  auto parent_t = parent->tensor<T, NDIMS + 1>();
  Eigen::DSizes<Eigen::DenseIndex, NDIMS + 1> slice_indices;
  slice_indices[0] = index;
  Eigen::DSizes<Eigen::DenseIndex, NDIMS + 1> slice_size;
  slice_size[0] = 1;
  for (size_t i = 1; i < slice_size.size(); ++i) {
    slice_size[i] = element_t.dimension(i - 1);
  }
  parent_t.slice(slice_indices, slice_size) = element_t.reshape(slice_size);
  return Status::OK();
}

template <int NDIMS>
Status HandleElementToLargerSliceWithRank(const Tensor& element, Tensor* parent,
                                          int index) {
#define HANDLE_TYPE(T)                                                   \
  case DataTypeToEnum<T>::value: {                                       \
    return HandleElementToLargerSlice<T, NDIMS>(element, parent, index); \
  }

  switch (element.dtype()) {
    TF_CALL_DATASET_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
    default:
      return errors::Unimplemented(
          "HandleElementToLargerSliceWithRank Unhandled data type: ",
          element.dtype());
  }
}

Status CopyElementToLargerSlice(const Tensor& element, Tensor* parent,
                                int index) {
  if (parent->dims() != element.dims() + 1) {
    return errors::Internal(
        "Mismatched ranks.  Element's rank is: ", element.dims(),
        " but element is meant to be a slice in output Tensor having rank: ",
        parent->dims(), " (should be: ", element.dims() + 1, ")");
  }

#define HANDLE_DIMS(NDIMS)                                                  \
  case NDIMS: {                                                             \
    TF_RETURN_IF_ERROR(                                                     \
        HandleElementToLargerSliceWithRank<NDIMS>(element, parent, index)); \
    return Status::OK();                                                    \
  }

  switch (element.dims()) {
    HANDLE_DIMS(0);
    HANDLE_DIMS(1);
    HANDLE_DIMS(2);
    HANDLE_DIMS(3);
    HANDLE_DIMS(4);
    HANDLE_DIMS(5);
#undef HANDLE_DIMS
    default:
      return errors::Unimplemented("CopyElementToLargerSlice Unhandled rank: ",
                                   element.dims());
  }
}

Status SetElementZero(Tensor* element, const Tensor& padding) {
#define HANDLE_TYPE(T)                                     \
  if (element->dtype() == DataTypeToEnum<T>::value) {      \
    element->flat<T>().setConstant(padding.scalar<T>()()); \
    return Status::OK();                                   \
  }
  TF_CALL_DATASET_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
  return errors::Unimplemented("SetElementZero Unhandled data type: ",
                               element->dtype());
}

}  // namespace batch_util
}  // namespace tensorflow
