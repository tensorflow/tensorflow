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
#ifndef TENSORFLOW_CORE_UTIL_BATCH_UTIL_H_
#define TENSORFLOW_CORE_UTIL_BATCH_UTIL_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace batch_util {

// Copies element into the index^th slice of parent (in the 0th dimension).
//
// NOTE(mrry): The `element` argument is taken by value. Use `std::move()`
// to move the `element` argument into this function, and the implementation
// may be able to optimize the copy to a move. This is particularly important
// for DT_STRING tensors.
Status CopyElementToSlice(Tensor element, Tensor* parent, int64 index);

// Copies the index^th slice of parent (in the 0th dimension) into element.
Status CopySliceToElement(const Tensor& parent, Tensor* element, int64 index);

// Copies 'num_slices' contiguous slices from 'src' tensor starting from index
// 'src_offset' into target tensor 'dst', and places them into slices
// starting from 'dst_offset'.
//
// This function requires 'src' and 'dst' to have compatible shapes. That is it
// requires cum_prod(src.shape[1:] == cum_prod(dst->shape[1:]). For example if
// source is of shape [x, 2, 1] and dst is a tensor of shape [y, 1, 2], this
// function can still proceed successfully.
Status CopyContiguousSlices(const Tensor& src, int64 src_offset,
                            int64 dst_offset, int64 num_slices, Tensor* dst);

// Copies the index^th slice of parent (in the 0th dimension) into element.
//
// NOTE(mrry): The implementation may be able to optimize the copy to a move.
// This is particularly important for DT_STRING tensors.
Status MaybeMoveSliceToElement(Tensor* parent, Tensor* element, int64 index);

// Zero-initializes the tensor `element` using the scalar stored in `padding`.
// Both `element` and `padding` must have matching `dtype`.
Status SetElementZero(Tensor* element, const Tensor& padding);

// Copies `element` into a (0th dimension) slice of `parent`, assuming
// the shape of `element` is strictly not larger along any axis than a
// slice.
Status CopyElementToLargerSlice(const Tensor& element, Tensor* parent,
                                int index);

}  // namespace batch_util
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_BATCH_UTIL_H_
