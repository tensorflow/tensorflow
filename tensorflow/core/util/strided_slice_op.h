/* Copyright 2016 Google Inc. All Rights Reserved.

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
#ifndef THIRD_PARTY_TENSORFLOW_CORE_UTIL_STRIDED_SLICE_OP_H_
#define THIRD_PARTY_TENSORFLOW_CORE_UTIL_STRIDED_SLICE_OP_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {

// Runs validation on the strided slice op parameters.
//
// Is a separate translation unit from the kernel so that:
// 1. The op's shape function can use it.
// 2. The code size is reduced vs templating this on the kernel's type.
Status ValidateStridedSliceOp(
    const Tensor& begin_tensor, const Tensor& end_tensor,
    const Tensor& strides_tensor, const TensorShape& input_shape,
    int32 begin_mask_spec, int32 end_mask_spec, const int32 ellipsis_mask,
    int32 new_axis_mask, int32 shrink_axis_mask, TensorShape* processing_shape,
    TensorShape* final_shape, bool* is_identity, bool* is_simple_slice,
    bool* slice_dim0, gtl::InlinedVector<int64, 4>* begin,
    gtl::InlinedVector<int64, 4>* end, gtl::InlinedVector<int64, 4>* strides);

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_UTIL_STRIDED_SLICE_OP_H_
