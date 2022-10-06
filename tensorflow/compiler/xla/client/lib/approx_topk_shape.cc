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

#include "tensorflow/compiler/xla/client/lib/approx_topk_shape.h"

#include <algorithm>

#include "tensorflow/compiler/xla/util.h"

// Used by rank 2+ operands
const uint64_t kTpuLaneTiling = 128;
// Used by rank 1 operands.
const uint64_t kTpuChunkTiling = 1024;

namespace xla {

inline uint32_t log2_floor(uint64_t value) {
  return value == 0 ? 0 : Log2Floor(value);
}

inline uint32_t log2_ceil(uint64_t value) {
  return value == 0 ? 0 : Log2Ceiling(value);
}

// LINT.IfChange
StatusOr<std::pair<int64_t, int64_t>> ApproxTopKReductionOutputSize(
    int64_t input_size, int64_t rank, int64_t top_k, float recall_target,
    bool aggregate_to_topk, int64_t input_size_override) {
  if (aggregate_to_topk) {
    return std::pair<int64_t, int64_t>(top_k, -1);
  }

  uint64_t tpu_tiling = rank == 1 ? kTpuChunkTiling : kTpuLaneTiling;

  if (input_size <= tpu_tiling) {
    return std::pair<int64_t, int64_t>(input_size, 0);
  }

  if (input_size_override >= 0) {
    if (input_size > input_size_override) {
      return InvalidArgument(
          "reduction_input_size_override: %d should be greater "
          "equals to operands[reduction_dim]: %d",
          input_size_override, input_size);
    }
  }
  uint64_t logical_input_size =
      input_size_override >= 0 ? input_size_override : input_size;

  // Reduce to the tiling size when k == 1.
  if (top_k == 1) {
    uint32_t log2_reduction =
        log2_ceil(CeilOfRatio(logical_input_size, tpu_tiling));
    return std::pair<int64_t, int64_t>(tpu_tiling, log2_reduction);
  }

  // Need to handle 1.0 explicitly, otherwise we would encounter division by
  // log(1.0) = 0 issue.
  if (recall_target == 1.0) {
    return std::pair<int64_t, int64_t>(input_size, 0);
  }

  if (recall_target <= 0. || recall_target > 1.0) {
    return InvalidArgument("recall_target should range in (0,1]");
  }

  // Given number of data points N, K for top-k elements, and W for the size of
  // the reduce window, let M = Ceil(N / W) be the number of windows. The
  // expected number of top-k elements that doesn't collide in windows is
  //
  //   K * ((M - 1) / M)^{K - 1}
  //
  // The recall of is the expected number of top-k elements divided by K
  //
  //   recall = ((M - 1) / M)^{K - 1}
  //          = (1 - 1/M)^{K - 1}
  //          = (1 - 1/M)^{-M * (K - 1)/(-M)}
  //          ~= EXP((1 - K) / M)    for large M
  //
  //   => M = (1 - K)/LOG(recall)
  uint64_t m = std::min<uint64_t>(
      std::max(
          static_cast<uint64_t>((1.0 - top_k) /
                                std::log(static_cast<double>(recall_target))),
          tpu_tiling),
      input_size);
  uint32_t log2_reduction = log2_floor(logical_input_size / m);
  if (log2_reduction == 0) {
    return std::pair<int64_t, int64_t>(input_size, 0);
  }

  // Do not reduce too much when the logical_input is too large.
  log2_reduction =
      std::min<uint32_t>(log2_reduction, log2_ceil(input_size / tpu_tiling));

  int64_t approx_output_size =
      CeilOfRatio<int64_t>(CeilOfRatio<int64_t>(input_size, tpu_tiling),
                           (1 << log2_reduction)) *
      tpu_tiling;

  return std::pair<int64_t, int64_t>(approx_output_size, log2_reduction);
}
// LINT.ThenChange(//tensorflow/core/ops/nn_ops.cc)

}  // namespace xla
