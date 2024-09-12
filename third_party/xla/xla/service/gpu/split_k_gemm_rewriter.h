/* Copyright 2023 The OpenXLA Authors.

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
#ifndef XLA_SERVICE_GPU_SPLIT_K_GEMM_REWRITER_H_
#define XLA_SERVICE_GPU_SPLIT_K_GEMM_REWRITER_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/matmul_utils.h"

namespace xla {
namespace gpu {

// Is there a non-empty suffix "s" of span such that product(s) % divisor = 0
// and for all t != s non-empty suffixes of s: d % product(t) = 0?
bool HasDivisibleSuffixAllowingSplit(absl::Span<int64_t const> span,
                                     int64_t divisor);

// Apply split K configuration from the tiling config to the fusion instruction:
// in addition to MakeDotComputationSplitKBatch on its computation add the
// necessary reduction after it.
absl::Status MakeDotSplitKBatch(HloInstruction* dot_fusion,
                                const TritonGemmConfig& config);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_SPLIT_K_GEMM_REWRITER_H_
