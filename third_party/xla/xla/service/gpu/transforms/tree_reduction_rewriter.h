/* Copyright 2020 The OpenXLA Authors.

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
#ifndef XLA_SERVICE_GPU_TRANSFORMS_TREE_REDUCTION_REWRITER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_TREE_REDUCTION_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Rewrites reductions in a way they can be implemented without atomics.
//
// Rule application: rewrite a single HLO reduce operation into two.
//
// Case 1: Row reduction, batched dimension is present, larger than
// Z-tiling size.
// -----------------------------------------------------------------
//
// Rewriting:
//
// f32[B] out = reduce(f32[A, B, C] input, dimensions={0, 2})
//
// Into:
//
// f32[A, B] tmp = reduce(f32[A, B, C] input, dimensions={2})
// f32[B] out = reduce(f32[A, B] tmp, dimensions={0})
//
// Case 2: Row reduction
// ------------------------------------------------------------------
//
// Let M be the thread tiling multiplied by the warp size.
// We go from (assuming C > M):
//
// f32[B] out = reduce(f32[A, B, C] input, dimensions={0, 2})
//
// to:
//
// f32[A, B, P] padded = pad(input) // Let P = ceil(C/M) * M.
// f32[A, B, Q, M] reshaped = bitcast(padded) // Let Q = ceil(C/M)
// f32[B, Q] inner_reduce = reduce(reshaped, dimensions={0, 3})
// f32[B] outer_reduce = reduce(inner_reduce, dimensions={1})
//
// Case 3: Column reduction
// -------------------------------------------------------------------
//
// Let T be the tiling size for the column reduction.
//
// We go from (assuming B > T):
//
// f32[A, C] out = reduce(f32[A, B, C] input, dimensions={1})
//
// to:
//
// f32[A, P, C] padded = pad(input) // Let P = ceil(B/T) * T.
// f32[A, Q, T, C] reshaped = bitcast(padded) // Let Q = ceil(B/T)
// f32[A, Q, C] inner_reduce = reduce(reshaped, dimensions={2})
// f32[A, C] outer_reduce = reduce(inner_reduce, dimensions={1})
//
class TreeReductionRewriter : public HloModulePass {
 public:
  explicit TreeReductionRewriter(se::GpuComputeCapability gpu_version)
      : gpu_version_(gpu_version) {}

  ~TreeReductionRewriter() override = default;
  absl::string_view name() const override { return "tree-reduction-rewriter"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::GpuComputeCapability gpu_version_;
};

}  // end namespace gpu
}  // end namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_TREE_REDUCTION_REWRITER_H_
