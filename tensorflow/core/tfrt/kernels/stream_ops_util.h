/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_KERNELS_STREAM_OPS_UTIL_H_
#define TENSORFLOW_CORE_TFRT_KERNELS_STREAM_OPS_UTIL_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace tfrt_stub {

// Unbatches `tensors` according to the step ids and returns a list of (step_id,
// unbatched_tensors) pairs.
//
// If `step_ids` is a scalar, each tensor in `tensors` is treated as if they are
// not batched and the entire tensor is associated with the single step id.
//
// If `step_ids` is a 1-D tensor, this tensor represents the step id of each
// example in the batch. Tensors in `tensors` are "unbatched" along the leading
// dimension according to the step id tensor and the unbatched tensors are
// associated with the corresponding step ids.
absl::StatusOr<std::vector<std::pair<int64_t, std::vector<tensorflow::Tensor>>>>
UnbatchStreamResults(const tensorflow::Tensor& step_ids,
                     absl::Span<const tensorflow::Tensor> tensors);

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_KERNELS_STREAM_OPS_UTIL_H_
