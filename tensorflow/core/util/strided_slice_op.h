/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_UTIL_STRIDED_SLICE_OP_H_
#define TENSORFLOW_CORE_UTIL_STRIDED_SLICE_OP_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {

struct StridedSliceShapeSpec {
  // Begin mask canonlized in dense form.
  int32 begin_dense_mask;
  // End mask canonlized in dense form.
  int32 end_dense_mask;
  // Shrink axis mask canonlized in dense form.
  int32 shrink_axis_dense_mask;
  // output_to_sparse_mapping[i] represents output[i]'s the corresponding dim
  // index in the begin_tensor. If
  // output_to_sparse_mapping[i] is -1, it means the dimension doesn't show up
  // in sparse_mapping.
  gtl::InlinedVector<int64, 4> output_to_sparse_mapping;
  // output_to_processing_mapping is similar to output_to_sparse_mapping, but
  // for processing shape.
  gtl::InlinedVector<int64, 4> output_to_processing_mapping;
  // processing_to_sparse_mapping[i] represents input_shape[i]'s corresponding
  // dim index in the begin_tensor.
  gtl::InlinedVector<int64, 4> processing_to_sparse_mapping;
};

// Runs validation on the strided slice op parameters.
//
// Is a separate translation unit from the kernel so that:
// 1. The op's shape function can use it.
// 2. The code size is reduced vs templating this on the kernel's type.
//
// Note that when input_shape is not fully specified, only <final_shape> and
// <processing_shape> are valid; <is_identity>, <is_simple_slice> and other
// output parameters will not be accurate.
//
// If <begin_tensor> or <end_tensor> are nullptr, <begin> and <end> will not be
// valid. In this case, <slice_dim0> and <is_identity> will be true only if a
// determination can be made based on the information given. A best effort is
// made to set <processing_shape> and <final_shape> based on <input_shape>, but
// some dimensions of <processing_shape> and/or <final_shape> may be unknown
// (-1). Any validation that can be done without complete information is
// performed.
//
Status ValidateStridedSliceOp(
    const Tensor* begin_tensor, const Tensor* end_tensor,
    const Tensor& strides_tensor, const PartialTensorShape& input_shape,
    int32 begin_mask_spec, int32 end_mask_spec, const int32 ellipsis_mask,
    int32 new_axis_mask, int32 shrink_axis_mask,
    PartialTensorShape* processing_shape, PartialTensorShape* final_shape,
    bool* is_identity, bool* is_simple_slice, bool* slice_dim0,
    gtl::InlinedVector<int64, 4>* begin, gtl::InlinedVector<int64, 4>* end,
    gtl::InlinedVector<int64, 4>* strides,
    StridedSliceShapeSpec* shape_spec = nullptr);

// Same as above, but the outputs are TensorShape, not PartialTensorShape
Status ValidateStridedSliceOp(
    const Tensor* begin_tensor, const Tensor* end_tensor,
    const Tensor& strides_tensor, const PartialTensorShape& input_shape,
    int32 begin_mask_spec, int32 end_mask_spec, const int32 ellipsis_mask,
    int32 new_axis_mask, int32 shrink_axis_mask, TensorShape* processing_shape,
    TensorShape* final_shape, bool* is_identity, bool* is_simple_slice,
    bool* slice_dim0, gtl::InlinedVector<int64, 4>* begin,
    gtl::InlinedVector<int64, 4>* end, gtl::InlinedVector<int64, 4>* strides,
    StridedSliceShapeSpec* shape_spec = nullptr);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_STRIDED_SLICE_OP_H_
