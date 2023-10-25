/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_CLIENT_LIB_APPROX_TOPK_SHAPE_H_
#define XLA_CLIENT_LIB_APPROX_TOPK_SHAPE_H_

#include <utility>

#include "xla/statusor.h"

namespace xla {

// Determine the output size of the reduction dimension. This is useful for jax
// abstract eval to determine the output size.
//
// input_size: Input size of the reduction dimension.
// rank: Rank of the input operand.
// top_k: Determines the k in top-k operation.
// recall_target: Valid range (0, 1]. User can trade-off quality and performance
//   with this knob.
// aggregate_to_topk: When true, sorts the set of approximate top-k elements and
//   only keep the final k elements on TPU. This option is useful when user
//   wanted to forward the approximate results to host and aggregate the results
//   on CPU for better throughput.
//
// Returns a pair of
//   1. Reduction output size
//   2. Reduction amount in log2 form.
//
// 2. is invalid and set to -1 when the approximate output is disabled, i.e.
//   top_k = 1 or aggregate_to_topk = true.
StatusOr<std::pair<int64_t, int64_t>> ApproxTopKReductionOutputSize(
    int64_t input_size, int64_t rank, int64_t top_k, float recall_target,
    bool aggregate_to_topk, int64_t input_size_override = -1);

}  // namespace xla

#endif  // XLA_CLIENT_LIB_APPROX_TOPK_SHAPE_H_
