/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GEMM_REWRITER_TRITON_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GEMM_REWRITER_TRITON_H_

#include <array>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/autotuning.pb.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_types.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Allowlist of unary elementwise operations supported by Triton GEMM codegen.
std::vector<HloOpcode> TritonSupportedUnaryElementwise(PrimitiveType);

// Allowlist of binary elementwise operations supported by Triton GEMM codegen.
std::vector<HloOpcode> TritonSupportedBinaryElementwise(PrimitiveType);

// Allowlist of ternary elementwise operations supported by Triton GEMM codegen.
std::vector<HloOpcode> TritonSupportedTernaryElementwise(PrimitiveType);

// Checks elementwise operation against all supported by Triton GEMM codegen.
bool IsTritonSupportedElementwise(HloOpcode, PrimitiveType);

// Apply split K configuration from the tiling to the fusion instruction:
// in addition to MakeDotComputationSplitKBatch on its computation add the
// necessary reduction after it.
Status MakeDotSplitKBatch(HloInstruction* dot_fusion,
                          const AutotuneResult::TritonGemmKey& tiling);

// Filters GEMMs which are better to handle using Triton.
bool IsTritonHandledGEMM(const HloInstruction&, GpuVersion gpu_version);

// Analysis of iteration of HLO shapes within a fusion around dot().
class DotFusionAnalysis {
 public:
  // Description of basic iteration: `count` elements separated by `stride`.
  struct IterationSpecFragment {
    int64_t stride;
    int64_t count;
    // Logical subfragments when this iteration is composed
    // of several HLO dimensions. Product of subfragments equals `count`.
    std::vector<int64_t> subfragments;
  };

  // Description of complex iteration over a sequence of several strides.
  // Describes a logically contiguous dimension of a tensor physically
  // separated into multiple fragments by other dimensions.
  using DimIterationSpec = std::vector<IterationSpecFragment>;

  // At most: contracting, non-contracting, split-K, another batch.
  static const int kMaxDimsPerTensor = 4;
  using TensorIterationSpec = std::array<DimIterationSpec, kMaxDimsPerTensor>;

  // Execute analysis of dot fusion computation.
  // split_k indicates whether this operation was converted to the split-K
  // form and tells the analysis how to interpret the batch dimensions.
  explicit DotFusionAnalysis(const HloComputation*, int64_t split_k = 1);

  // A scope is an HLO graph that can be tiled efficiently using same or
  // compatible tile shapes on all operations. GEMM fusion has 3 scopes
  // defined by left operand, right operand and output.
  enum class Scope { LHS = 0, RHS = 1, OUTPUT = 2 };

  // Scope -> HLO -> dot dimension number -> iteration spec at the HLO's output.
  const DimIterationSpec* IterSpec(Scope scope, const HloInstruction*,
                                   int dimension) const;
  // Parameter HLO instructions used in a scope of `dot`.
  const absl::flat_hash_set<const HloInstruction*>& ScopeParameters(
      const Scope scope) const {
    return parameters_.at(scope);
  }

 private:
  absl::flat_hash_map<
      Scope, absl::flat_hash_map<const HloInstruction*, TensorIterationSpec>>
      iter_specs_;
  // HLO computation parameters per scope.
  absl::flat_hash_map<Scope, absl::flat_hash_set<const HloInstruction*>>
      parameters_;
};

// Rewrite compatible dot() calls into custom calls with fused computations
// that target Triton-based matmul emitter.
class GemmRewriterTriton : public HloModulePass {
 public:
  explicit GemmRewriterTriton(GpuVersion gpu_version)
      : gpu_version_(gpu_version) {}
  absl::string_view name() const override { return "triton-gemm-rewriter"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  GpuVersion gpu_version_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GEMM_REWRITER_TRITON_H_
