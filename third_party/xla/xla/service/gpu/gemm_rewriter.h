/* Copyright 2019 The OpenXLA Authors.

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
#ifndef XLA_SERVICE_GPU_GEMM_REWRITER_H_
#define XLA_SERVICE_GPU_GEMM_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// cuBLAS GEMM in the most general form can run the following operation:
//
// (kAdd
//    (kMultiply (kDot A B) alpha)
//    (kMultiply C beta))
//
// where A, B, C are matrices or vectors and `alpha` and `beta` are host
// constants. In matrix-vector multiplication, one operand must be a matrix and
// the other must be a vector. The additional requirement is that C has no other
// users (otherwise, it does not make sense to fuse it inside the custom call).
//
// Both multiplication and addition can be avoided (equivalent to setting
// `alpha` to one and `beta` to zero).
//
// This pass pattern-matches the most general form of this instruction
// (we assume transposes are already folded), and rewrites it into a custom call
// where (A, B, C) are three operands respectively, and `alpha` and `beta` are
// stored in the backend config.
class GemmRewriter : public HloModulePass {
 public:
  explicit GemmRewriter(se::GpuComputeCapability gpu_version);
  absl::string_view name() const override { return "cublas-gemm-rewriter"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::GpuComputeCapability gpu_version_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GEMM_REWRITER_H_
