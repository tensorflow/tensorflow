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
#ifndef XLA_SERVICE_GPU_TRANSFORMS_GEMM_REWRITER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_GEMM_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"

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

struct GemmRewriterOptions {
  // The DType of the GEMM to rewrite.
  enum class DType { kFp8Only, kNonFp8Only };
  DType dtype = DType::kNonFp8Only;

  // Disabling bias prevents using the `beta * C` term the GEMM, which can
  // remove dependencies between multiple matrix multiplications. This, in
  // turn, can improve the performance of overall computation by allowing
  // multiple GEMMs to be scheduled in parallel.
  //
  // As an example, consider the following computation: `(A * A) + (B * B)`.
  // With bias enabled, the `GemmRewriter` will emit the following GEMMs:
  //
  // AA := GEMM(A * A)
  // ROOT := GEMM(B * B + AA)
  //
  // Because the second GEMM depends on the first, they cannot be scheduled in
  // parallel. Instead, with bias disabled, the `GemmRewriter` will emit the
  // following:
  //
  // AA := GEMM(A * A)
  // BB := GEMM(B * B)
  // ROOT := AA + BB
  //
  // In this case, the two GEMMs can be scheduled in parallel.
  enum class BiasMode { kBias, kNoBias };
  BiasMode bias_mode = BiasMode::kBias;
};

class GemmRewriter : public HloModulePass {
 public:
  GemmRewriter(se::GpuComputeCapability gpu_version,
               se::SemanticVersion toolkit_version,
               GemmRewriterOptions options = {});
  absl::string_view name() const override { return "cublas-gemm-rewriter"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::GpuComputeCapability gpu_version_;
  se::SemanticVersion toolkit_version_;
  GemmRewriterOptions options_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_GEMM_REWRITER_H_
