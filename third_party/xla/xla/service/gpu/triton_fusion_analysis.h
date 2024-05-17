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
#ifndef XLA_SERVICE_GPU_TRITON_FUSION_ANALYSIS_H_
#define XLA_SERVICE_GPU_TRITON_FUSION_ANALYSIS_H_

// This file contains TritonFusionAnalysis and FusionContext.

#include <map>
#include <string>

#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/triton_tiling_propagation.h"
#include "xla/status.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Analysis of tensor iteration orders within tiled fusions.
class TritonFusionAnalysis {
  absl::Status ExecuteForDotFusion(const HloInstruction& dot, int split_k);
  absl::Status ExecuteForSoftmaxFusion(const HloInstruction& root);

 public:
  // Execute the analysis of a fusion computation.
  // `split_k` indicates whether this operation was converted to the split-K
  // form and tells the analysis how to interpret the batch dimensions.
  static absl::StatusOr<TritonFusionAnalysis> Execute(
      const HloComputation& computation, int split_k = 1);

  // Execute the analysis of a produce-consumer fusion. Returns OkStatus, if the
  // analysis can find a valid tiling for the producer-consumer fusion.
  // `split_k` indicates whether this operation was converted to the split-K
  // form and tells the analysis how to interpret the batch dimensions.
  static absl::Status ExecuteForProducerConsumer(const HloInstruction& producer,
                                                 const HloInstruction& consumer,
                                                 int split_k = 1);

  // A scope is an HLO graph that can be tiled efficiently using same or
  // compatible tile shapes on all operations. GEMM fusion has 3 or 4 scopes
  // defined by left operand, right operand, optional meta (third operand) and
  // output.
  enum class Scope { LHS = 0, RHS = 1, META = 2, OUTPUT = 3 };

  using IterationSpecByInstructionMap =
      ConstHloInstructionMap<TensorIterationSpec>;
  using IterationSpecByInstructionByScopeMap =
      std::map<Scope, IterationSpecByInstructionMap>;

  // Every parameter requires a separate piece of shared memory for asynchronous
  // loads. Multiple parameters are approximately equivalent to multiple
  // pipeline stages.
  // Note: This has been tuned specifically for GEMMs, where pipelining with
  // more than 4 stages has been shown to rarely be practical. This limitation
  // is not necessarily applicable to other operations.
  // Note: The limit doesn't apply to the epilogue of the fusion.
  static constexpr int kMaxParameterPerDotOperand = 4;

  // Scope -> HLO -> dot dimension number -> iteration spec at the HLO's output.
  const TensorIterationSpec::DimIterationSpec* IterSpec(Scope scope,
                                                        const HloInstruction*,
                                                        int dimension) const;
  // Parameter HLO instructions used in a scope of `dot`.
  const ConstHloInstructionSet& ScopeParameters(const Scope scope) const {
    return parameters_.at(scope);
  }

  // Returns the given instruction's scope, if there is no scope, returns
  // nullopt instead.
  std::optional<Scope> QueryInstructionScope(const HloInstruction& hlo) const;

  std::string ToString() const;

 private:
  IterationSpecByInstructionByScopeMap iter_specs_;
  // HLO computation parameters per scope.
  std::map<Scope, ConstHloInstructionSet> parameters_;
};

// The details of the Triton fusion / tiling propagation are in a separate
// namespace to avoid littering the xla::gpu namespace.
namespace triton_fusion {
class FusionContext {
  FusionContext(HeroProperties properties, Requirements requirements)
      : properties_(properties), requirements_(requirements) {}

 public:
  // Create fusion context from a dot operand according to
  // the currently supported configurations.
  static absl::StatusOr<FusionContext> FromDotOperand(const HloInstruction& dot,
                                                      int operand_number,
                                                      int split_k = 1);

  // Create fusion context from dot's output.
  static FusionContext FromDotOutput(const HloInstruction& dot, int split_k,
                                     DotRequirements requirements);

  static FusionContext FromSoftmaxRoot(const HloInstruction&);

  // Add dimension orders from `update` to `dim_orders_` and update
  // `requirements_` if all of them are compatible.
  bool CombineDimOrdersAndReqs(const DimOrdersAndReqs& update);

  // Propagate dimension orders in consumer->producer direction starting at
  // `origin` with output `origin_dim_order` till parameters of the
  // computation. Store the found parameters and their iteration specs.
  absl::Status PropagateDimensionOrdersToParameters(
      const HloInstruction& origin, ConstHloInstructionSet& parameters,
      ConstHloInstructionMap<TensorIterationSpec>& iter_specs);

  const HeroProperties& hero_properties() const { return properties_; }
  const DimOrderMap& dim_orders() const { return dim_orders_; }
  const Requirements& requirements() const { return requirements_; }

 private:
  const HeroProperties properties_;
  Requirements requirements_;
  DimOrderMap dim_orders_;
};

}  // namespace triton_fusion

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRITON_FUSION_ANALYSIS_H_
