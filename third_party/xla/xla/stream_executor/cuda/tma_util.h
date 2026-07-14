/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_TMA_UTIL_H_
#define XLA_STREAM_EXECUTOR_CUDA_TMA_UTIL_H_

#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/gpu/tma_metadata.h"

namespace stream_executor::gpu {

absl::StatusOr<CUtensorMapDataType> GetTensorMapDataType(int element_size);

CUtensorMapSwizzle GetTensorMapSwizzle(TmaDescriptor::TmaSwizzle swizzle);

CUtensorMapL2promotion GetTensorMapL2Promotion(
    TmaDescriptor::TmaL2Promotion l2_promotion);

CUtensorMapFloatOOBfill GetTensorMapFloatOOBFill(
    TmaDescriptor::TmaFloatOobFill oob_fill);

CUtensorMapInterleave GetTensorMapInterleave(
    TmaDescriptor::TmaInterleave interleave);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_TMA_UTIL_H_
