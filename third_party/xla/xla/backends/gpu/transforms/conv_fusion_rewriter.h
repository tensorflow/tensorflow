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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_CONV_FUSION_REWRITER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_CONV_FUSION_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/gpu/backend_configs.pb.h"

namespace xla {
namespace gpu {

// This pass identifies HLO convolution instructions that can be fused with
// their operands (prologue) and users (epilogue) for the cuDNN GPU backend.
//
// It uses a depth-first search (DFS) to explore the graph downstream from
// each convolution, identifying "pointwise" or elementwise operations and
// reductions that are compatible with cuDNN's fusion capabilities.
//
// The rewriter enforces several constraints to ensure the resulting fusion is
// supported by cuDNN:
//   1. All internal operations must be supported pointwise ops.
//   2. Fusions can have a maximum of 2 output tensors.
//   3. If there are 2 outputs, exactly one must be a reduction.
//   4. No cycles may be introduced in the HLO graph by the fusion.
//
// When a compatible subgraph is found, it is replaced with an HloFusion
// instruction. The instruction can be emitted by cuDNN and exectuted using
// cuDNN's graph API.

class ConvFusionRewriter : public HloModulePass {
 public:
  explicit ConvFusionRewriter() = default;

  absl::string_view name() const override { return "conv-fusion-rewriter"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_CONV_FUSION_REWRITER_H_
