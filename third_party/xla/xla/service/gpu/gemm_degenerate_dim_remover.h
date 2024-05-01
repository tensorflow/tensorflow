/* Copyright 2024 The OpenXLA Authors.

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
#ifndef XLA_SERVICE_GPU_GEMM_DEGENERATE_DIM_REMOVER_H_
#define XLA_SERVICE_GPU_GEMM_DEGENERATE_DIM_REMOVER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Rewrite a gemm with a degenerate dimension to a matrix-vector multiplication.
// For example, [m x n] @ [n x 1] is rewritten to [m x n] @ [n], and [n x 1]
// @ [m x n] is rewritten to [n] @ [m x n].
//
// The degenerate dimension is introduced by GemvRewriter, we should remove it
// after GemmFusion is run.
class GemmDegenerateDimRemover : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "gemm-degenerate-dim-remover";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GEMM_DEGENERATE_DIM_REMOVER_H_
