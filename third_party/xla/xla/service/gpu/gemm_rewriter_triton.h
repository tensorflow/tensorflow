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
#ifndef XLA_SERVICE_GPU_GEMM_REWRITER_TRITON_H_
#define XLA_SERVICE_GPU_GEMM_REWRITER_TRITON_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/service/instruction_fusion.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Tells if f(a+b) == f(a) + f(b).
bool IsDistributiveOverAddition(const HloInstruction& hlo);

// Allowlist of unary elementwise operations supported by Triton GEMM codegen.
std::vector<HloOpcode> TritonSupportedUnaryElementwise(PrimitiveType);

// Allowlist of binary elementwise operations supported by Triton GEMM codegen.
std::vector<HloOpcode> TritonSupportedBinaryElementwise(PrimitiveType);

// Allowlist of ternary elementwise operations supported by Triton GEMM codegen.
std::vector<HloOpcode> TritonSupportedTernaryElementwise(PrimitiveType);

// Data types that are supported by the Triton emitters.
bool IsTritonSupportedDataType(PrimitiveType, se::GpuComputeCapability);

// Checks elementwise operation against all supported by Triton GEMM codegen.
bool IsTritonSupportedElementwise(HloOpcode, PrimitiveType);

// Filters GEMMs which can be handled using Triton.
FusionDecision CanTritonHandleGEMM(const HloInstruction&,
                                   se::GpuComputeCapability gpu_version);

// Filters GEMMs which are better to handle using Triton.
bool ShouldTritonHandleGEMM(HloInstruction&,
                            se::GpuComputeCapability gpu_version);

class TensorIterationSpec {
 public:
  // Description of basic iteration: `count` elements separated by `stride`.
  struct IterationSpecFragment {
    int64_t stride;
    int64_t count;
    int64_t slice_start;
    int64_t slice_limit;
    // Logical subfragments when this iteration is composed
    // of several HLO dimensions.
    std::vector<int64_t> subfragments;

    bool is_sliced() const { return count != slice_limit - slice_start; }
    bool operator!=(const IterationSpecFragment& other) const;
    std::string ToString() const;
  };
  // Description of complex iteration over a sequence of several strides.
  // Describes a logically contiguous dimension of a tensor physically
  // separated into multiple fragments by other dimensions.
  using DimIterationSpec = std::vector<IterationSpecFragment>;

  using StorageType = absl::flat_hash_map<int, DimIterationSpec>;
  const DimIterationSpec& operator[](const int dimension) const {
    return dim_iteration_specs_.at(dimension);
  }
  DimIterationSpec& operator[](const int dimension) {
    return dim_iteration_specs_[dimension];
  }
  const StorageType& Storage() const { return dim_iteration_specs_; }
  void RemoveEmptyDimensions() {
    absl::erase_if(dim_iteration_specs_,
                   [](const auto& it) { return it.second.empty(); });
  }

  // Compares physical layouts of tensors ignoring subfragments of dimensions.
  bool operator==(const TensorIterationSpec& other) const;

  std::string ToString() const;

 private:
  StorageType dim_iteration_specs_;
};

// Analysis of tensor iteration orders within tiled fusions.
class TritonFusionAnalysis {
  Status ExecuteForDotFusion(const HloInstruction& dot, int split_k);
  Status ExecuteForSoftmaxFusion(const HloInstruction& root);

 public:
  // Execute the analysis of a fusion computation.
  // `split_k` indicates whether this operation was converted to the split-K
  // form and tells the analysis how to interpret the batch dimensions.
  static StatusOr<TritonFusionAnalysis> Execute(
      const HloComputation& computation, int split_k = 1);

  // A scope is an HLO graph that can be tiled efficiently using same or
  // compatible tile shapes on all operations. GEMM fusion has 3 scopes
  // defined by left operand, right operand and output.
  enum class Scope { LHS = 0, RHS = 1, OUTPUT = 2 };

  using IterationSpecByInstructionMap =
      ConstHloInstructionMap<TensorIterationSpec>;
  using IterationSpecByInstructionByScopeMap =
      std::map<Scope, IterationSpecByInstructionMap>;

  // Every parameter requires a separate piece of shared memory for asynchronous
  // loads. Multiple parameters are approximately equivalent to multiple
  // pipeline stages.
  // Note: this has been tuned specifically for GEMMs, where pipelining with
  // more than 4 stages has been shown to rarely be practical. This limitation
  // is not necessarily applicable to other operations.
  static constexpr int kMaxParameterPerDotScope = 4;

  // Scope -> HLO -> dot dimension number -> iteration spec at the HLO's output.
  const TensorIterationSpec::DimIterationSpec* IterSpec(Scope scope,
                                                        const HloInstruction*,
                                                        int dimension) const;
  // Parameter HLO instructions used in a scope of `dot`.
  const ConstHloInstructionSet& ScopeParameters(const Scope scope) const {
    return parameters_.at(scope);
  }

  std::string ToString() const;

 private:
  IterationSpecByInstructionByScopeMap iter_specs_;
  // HLO computation parameters per scope.
  std::map<Scope, ConstHloInstructionSet> parameters_;
};

// Rewrite compatible dot() calls into custom calls with fused computations
// that target Triton-based matmul emitter.
class GemmRewriterTriton : public HloModulePass {
 public:
  explicit GemmRewriterTriton(se::GpuComputeCapability gpu_version)
      : gpu_version_(gpu_version) {}
  absl::string_view name() const override { return "triton-gemm-rewriter"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::GpuComputeCapability gpu_version_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GEMM_REWRITER_TRITON_H_
