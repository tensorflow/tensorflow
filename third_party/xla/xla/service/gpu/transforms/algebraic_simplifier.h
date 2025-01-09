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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_ALGEBRAIC_SIMPLIFIER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_ALGEBRAIC_SIMPLIFIER_H_

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"

namespace xla::gpu {

class GpuAlgebraicSimplifierVisitor : public AlgebraicSimplifierVisitor {
 public:
  explicit GpuAlgebraicSimplifierVisitor(
      const AlgebraicSimplifierOptions& options,
      se::GpuComputeCapability compute_capability,
      AlgebraicSimplifier* simplifier)
      : AlgebraicSimplifierVisitor(options, simplifier),
        compute_capability_(std::move(compute_capability)) {}

  absl::Status HandleAdd(HloInstruction* add) override;

  bool ShouldStrengthReduceDotToReduce(const HloInstruction* hlo) override;

 private:
  // Returns true if the dot precision config is supported by simplifier.
  bool SupportedDotPrecisionConfig(const PrecisionConfig& config) override;

  // Makes algorithm specific set of instructions for multiply with precision
  // algorithm in mind. In the trivial case it returns just multiply.
  // For x3 or x6 algorithms it adds the parameters split instructions and the
  // corresponding multiply instructions.
  absl::StatusOr<HloInstruction*> MakeMultiplyForPrecisionAlgorithm(
      HloInstruction* dot, HloInstruction* lhs, HloInstruction* rhs) override;

  // Try to convert add(broadcast(const_0), add(broadcast(const_1), conv(...)))
  // into add(broadcast(add(const_0, const_1)), conv(...)) and return true if
  // successful. The particular sink happens only when enable_sink_broadcast is
  // true and the broadcast shapes and dimensions match. The sink only happens
  // when following a convolution to avoid having a side input when the
  // instructions are fused to cudnnConvolutionBiasActivationForward later.
  absl::StatusOr<bool> TryToSinkBroadcastOperandsOfChainedAdds(
      HloInstruction* add);

  se::GpuComputeCapability compute_capability_;
};

class GpuAlgebraicSimplifier : public AlgebraicSimplifier {
 public:
  explicit GpuAlgebraicSimplifier(const AlgebraicSimplifierOptions& options,
                                  se::GpuComputeCapability compute_capability)
      : AlgebraicSimplifier(options),
        compute_capability_(std::move(compute_capability)) {}

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(HloModule* module,
                           const absl::flat_hash_set<absl::string_view>&
                               execution_threads) override {
    XLA_VLOG_LINES(
        2, "GpuAlgebraicSimplifier::Run(), before:\n" + module->ToString());
    bool changed = false;
    GpuAlgebraicSimplifierVisitor visitor(options_, compute_capability_, this);
    for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
      if (visitor.Run(comp, options_, this)) {
        changed = true;
      }
    }
    XLA_VLOG_LINES(
        2, "GpuAlgebraicSimplifier::Run(), after:\n" + module->ToString());
    return changed;
  }

 private:
  se::GpuComputeCapability compute_capability_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_TRANSFORMS_ALGEBRAIC_SIMPLIFIER_H_
