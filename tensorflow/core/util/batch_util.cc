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

Status ValidateInput(const Tensor& parent, const Tensor& element, int64 index) {
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
Status HandleElementToSlice(Tensor element, Tensor* parent, int64 index,
                            bool /* can_move */) {
  parent->flat_outer_dims<T>().chip(index, 0) = element.flat<T>();
  return Status::OK();
}

template <>
Status HandleElementToSlice<string>(Tensor element, Tensor* parent, int64 index,
                                    bool can_move) {
  auto parent_as_matrix = parent->flat_outer_dims<string>();
  auto element_flat = element.flat<string>();
  if (can_move) {
    for (int64 i = 0; i < element.NumElements(); ++i) {
      parent_as_matrix(index, i) = std::move(element_flat(i));
    }
  } else {
    parent_as_matrix.chip(index, 0) = element_flat;
  }
  return Status::OK();
}

template <>
Status HandleElementToSlice<Variant>(Tensor element, Tensor* parent,
                                     int64 index, bool can_move) {
  auto parent_as_matrix = parent->flat_outer_dims<Variant>();
  auto element_flat = element.flat<Variant>();
  if (can_move) {
    for (int64 i = 0; i < element.NumElements(); ++i) {
      parent_as_matrix(index, i) = std::move(element_flat(i));
    }
  } else {
    parent_as_matrix.chip(index, 0) = element_flat;
  }
  return Status::OK();
}

// TODO(b/78245576): Consider removing this overload.
template <typename T>
void HandleSliceToElement(const Tensor& parent, Tensor* element, int64 index) {
  element->flat<T>() = parent.flat_outer_dims<T>().chip(index, 0);
}

template <typename T>
void HandleSliceToElement(Tensor* parent, Tensor* element, int64 index,
                          bool can_move) {
  element->flat<T>() = parent->flat_outer_dims<T>().chip(index, 0);
}

template <>
void HandleSliceToElement<string>(Tensor* parent, Tensor* element, int64 index,
                                  bool can_move) {
  auto parent_as_matrix = parent->flat_outer_dims<string>();
  auto element_flat = element->flat<string>();
  if (can_move) {
    for (int64 i = 0; i < element->NumElements(); ++i) {
      element_flat(i) = std::move(parent_as_matrix(index, i));
    }
  } else {
    element_flat = parent_as_matrix.chip(index, 0);
  }
}

template <>
void HandleSliceToElement<Variant>(Tensor* parent, Tensor* element, int64 index,
                                   bool can_move) {
  auto parent_as_matrix = parent->flat_outer_dims<Variant>();
  auto element_flat = element->flat<Variant>();
  if (can_move) {
    for (int64 i = 0; i < element->NumElements(); ++i) {
      element_flat(i) = std::move(parent_as_matrix(index, i));
    }
  } else {
    element_flat = parent_as_matrix.chip(index, 0);
  }
}

}  // namespace

// Copies element into the index^th slice of parent (in the 0th dimension).
Status CopyElementToSlice(Tensor element, Tensor* parent, int64 index) {
  TF_RETURN_IF_ERROR(ValidateInput(*parent, element, index));

  bool can_move = element.RefCountIsOne();
#define HANDLE_TYPE(T)                                                \
  case DataTypeToEnum<T>::value: {                                    \
    return HandleElementToSlice<T>(std::move(element), parent, index, \
                                   can_move);                         \
  }

  switch (element.dtype()) {
    TF_CALL_ALL_TYPES(HANDLE_TYPE);
    TF_CALL_QUANTIZED_TYPES(HANDLE_TYPE);
    TF_CALL_uint32(HANDLE_TYPE);
    TF_CALL_uint64(HANDLE_TYPE);
#undef HANDLE_TYPE
    default:
      return errors::Unimplemented("CopyElementToSlice Unhandled data type: ",
                                   element.dtype());
  }
}

// Copies the index^th slice of parent (in the 0th dimension) into element.
Status CopySliceToElement(const Tensor& parent, Tensor* element, int64 index) {
  TF_RETURN_IF_ERROR(ValidateInput(parent, *element, index));

#define HANDLE_TYPE(T)                               \
  case DataTypeToEnum<T>::value: {                   \
    HandleSliceToElement<T>(parent, element, index); \
    return Status::OK();                             \
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

// Copies the index^th slice of parent (in the 0th dimension) into element.
//
// NOTE(mrry): The implementation may be able to optimize the copy to a move.
// This is particularly important for DT_STRING tensors.
Status MaybeMoveSliceToElement(Tensor* parent, Tensor* element, int64 index) {
  TF_RETURN_IF_ERROR(ValidateInput(*parent, *element, index));
  bool can_move = parent->RefCountIsOne();

#define HANDLE_TYPE(T)                                         \
  case DataTypeToEnum<T>::value: {                             \
    HandleSliceToElement<T>(parent, element, index, can_move); \
    return Status::OK();                                       \
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
