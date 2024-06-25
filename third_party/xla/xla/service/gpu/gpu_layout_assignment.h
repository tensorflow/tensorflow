/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_GPU_LAYOUT_ASSIGNMENT_H_
#define XLA_SERVICE_GPU_GPU_LAYOUT_ASSIGNMENT_H_

#include <cstdint>
#include <initializer_list>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/computation_layout.h"
#include "xla/service/layout_assignment.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"

namespace xla {
namespace gpu {

// GPU-specific layout assignment pass which preassigns layouts to satisfy
// layout constraints for operands and results of library calls.
class GpuLayoutAssignment : public LayoutAssignment {
 public:
  explicit GpuLayoutAssignment(
      ComputationLayout* entry_computation_layout,
      const se::GpuComputeCapability& gpu_version,
      const se::dnn::VersionInfo& dnn_version,
      ChannelLayoutConstraints* channel_constraints = nullptr)
      : LayoutAssignment(entry_computation_layout, channel_constraints),
        gpu_version_(gpu_version),
        dnn_version_(dnn_version) {}
  ~GpuLayoutAssignment() override = default;

 protected:
  absl::Status AddBackendConstraints(LayoutConstraints* constraints) override;

 private:
  absl::Status AddBackendConstraintsToDnnConvCustomCall(
      HloCustomCallInstruction* instr, LayoutConstraints* constraints);

  // dim_groups are ordered from major to minor dimensions.
  absl::Status SetOperandMajorToMinorLayout(
      const HloInstruction* instruction, int64_t operand,
      std::initializer_list<absl::Span<const int64_t>> dim_groups);

  absl::Status SetDotOperandLayout(const HloInstruction* instruction,
                                   int64_t operand,
                                   absl::Span<const int64_t> batch_dims,
                                   absl::Span<const int64_t> row_dims,
                                   absl::Span<const int64_t> col_dims);

  absl::Status SetDotLayout(const HloInstruction* instruction,
                            LayoutConstraints* constraints);

  bool PropagateReductionLayoutToOperand(const HloInstruction* user) override;

  bool InstructionCanChangeLayoutInstance(
      const HloInstruction* instruction) override;

  const se::GpuComputeCapability gpu_version_;
  const se::dnn::VersionInfo dnn_version_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_LAYOUT_ASSIGNMENT_H_
