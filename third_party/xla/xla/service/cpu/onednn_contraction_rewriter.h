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

#ifndef XLA_SERVICE_CPU_ONEDNN_CONTRACTION_REWRITER_H_
#define XLA_SERVICE_CPU_ONEDNN_CONTRACTION_REWRITER_H_
#if defined(INTEL_MKL)

#include <optional>

#include "absl/algorithm/container.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/cpu/onednn_convolution.h"
#include "xla/service/cpu/onednn_matmul.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace cpu {

// This pass pattern-matches HLO Dot and Convolution instructions and rewrites
// them into custom calls.
class OneDnnContractionRewriter : public HloModulePass {
 public:
  OneDnnContractionRewriter(int intra_op_parallelism,
                            const tsl::thread::ThreadPool* compile_threadpool)
      : intra_op_parallelism_(intra_op_parallelism),
        compile_threadpool_(compile_threadpool) {}
  OneDnnContractionRewriter() = default;
  absl::string_view name() const override {
    return "onednn-contraction-rewriter";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  static bool ShouldRewriteDot(const HloInstruction* dot_instr,
                               bool before_layout_assignment = false);
  static bool ShouldRewriteConv(const HloInstruction* conv_instr);
  static bool ShouldRewriteInstr(const HloInstruction* instr,
                                 bool before_layout_assignment = false) {
    return ShouldRewriteDot(instr, before_layout_assignment) ||
           ShouldRewriteConv(instr);
  }

 private:
  int intra_op_parallelism_;
  const tsl::thread::ThreadPool* compile_threadpool_;
};

using OneDnnContractionVariant =
    std::variant<PrimitiveTrait<kOnednnConvConfig>,
                 PrimitiveTrait<kOnednnMatmulConfig>>;

template <BackendConfigOneofCase config>
struct PrimitiveTrait<config, OneDnnFusionConfig*> {
  static OneDnnFusionConfig* GetTransformationConfig(
      typename PrimitiveTrait<config>::pointer_type kernel_config) {
    return kernel_config->mutable_fusions();
  }
};

template <BackendConfigOneofCase config>
struct PrimitiveTrait<config, OneDnnOptimizationConfig*> {
  static OneDnnOptimizationConfig* GetTransformationConfig(
      typename PrimitiveTrait<config>::pointer_type kernel_config) {
    return kernel_config->mutable_optimization_config();
  }
};

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL
#endif  // XLA_SERVICE_CPU_ONEDNN_CONTRACTION_REWRITER_H_
