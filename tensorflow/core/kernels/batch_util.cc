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

#include "tensorflow/core/kernels/batch_util.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace batch_util {

namespace {

// Copies element into the index^th slice of parent (in the 0th dimension).
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

}  // namespace

Status CopyElementToSlice(Tensor element, Tensor* parent, int64 index) {
  if (element.NumElements() != (parent->NumElements() / parent->dim_size(0))) {
    TensorShape chip_shape = parent->shape();
    chip_shape.RemoveDim(0);
    return errors::InvalidArgument(
        "HandleElementToSlice Cannot copy slice: number of elements does "
        "not match. Shapes are: [element]: ",
        element.shape().DebugString(),
        ", [parent slice]: ", chip_shape.DebugString());
  }
  bool can_move = element.RefCountIsOne();
#define HANDLE_TYPE(T)                                                \
  case DataTypeToEnum<T>::value: {                                    \
    return HandleElementToSlice<T>(std::move(element), parent, index, \
                                   can_move);                         \
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

}  // namespace batch_util
}  // namespace tensorflow
