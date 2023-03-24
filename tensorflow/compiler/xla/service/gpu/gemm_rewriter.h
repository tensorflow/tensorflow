/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GEMM_REWRITER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GEMM_REWRITER_H_

#include <optional>

#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// cuBLAS GEMM in the most general form can run the following operation:
//
// (kAdd
//    (kMultiply (kDot A B) alpha)
//    (kMultiply C beta))
//
// where A, B, C are matrixes and `alpha` and `beta` are host constants.
// The additional requirement is that C has no other users (otherwise,
// it does not make sense to fuse it inside the custom call).
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
  explicit GemmRewriter(se::CudaComputeCapability cuda_compute_capability);
  absl::string_view name() const override { return "cublas-gemm-rewriter"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::CudaComputeCapability cuda_compute_capability_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GEMM_REWRITER_H_
