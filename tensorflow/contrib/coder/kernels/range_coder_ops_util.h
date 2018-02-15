/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_CODER_KERNELS_RANGE_CODER_OPS_UTIL_H_
#define TENSORFLOW_CONTRIB_CODER_KERNELS_RANGE_CODER_OPS_UTIL_H_

#include <vector>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
// The shapes are simplified to reduce indexing cost.
Status MergeAxes(const TensorShape& broadcast_shape,
                 const TensorShape& storage_shape,
                 std::vector<int64>* merged_broadcast_shape_pointer,
                 std::vector<int64>* merged_storage_shape_pointer);
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_CODER_KERNELS_RANGE_CODER_OPS_UTIL_H_
