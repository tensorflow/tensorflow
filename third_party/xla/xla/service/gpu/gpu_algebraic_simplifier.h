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

#ifndef XLA_SERVICE_GPU_GPU_ALGEBRAIC_SIMPLIFIER_H_
#define XLA_SERVICE_GPU_GPU_ALGEBRAIC_SIMPLIFIER_H_

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/hlo_pass_interface.h"
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

  bool ShouldStrengthReduceDotToReduce(const HloInstruction* hlo) override;

 private:
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

#endif  // XLA_SERVICE_GPU_GPU_ALGEBRAIC_SIMPLIFIER_H_
