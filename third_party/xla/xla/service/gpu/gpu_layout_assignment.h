/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/computation_layout.h"
#include "xla/service/layout_assignment.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/status.h"

namespace xla {
namespace gpu {

// GPU-specific layout assignment pass which preassigns layouts to satisfy
// layout constraints for operands and results of library calls.
class GpuLayoutAssignment : public LayoutAssignment {
 public:
  explicit GpuLayoutAssignment(
      ComputationLayout* entry_computation_layout,
      se::StreamExecutor* stream_executor,
      ChannelLayoutConstraints* channel_constraints = nullptr)
      : LayoutAssignment(entry_computation_layout, channel_constraints),
        stream_executor_(stream_executor) {}
  ~GpuLayoutAssignment() override = default;

 protected:
  Status AddBackendConstraints(LayoutConstraints* constraints) override;

 private:
  Status AddBackendConstraintsToDnnConvCustomCall(
      HloCustomCallInstruction* instr, LayoutConstraints* constraints);

  // dim_groups are ordered from major to minor dimensions.
  Status SetOperandMajorToMinorLayout(
      const HloInstruction* instruction, int64_t operand,
      std::initializer_list<absl::Span<const int64_t>> dim_groups);

  Status SetDotOperandLayout(const HloInstruction* instruction, int64_t operand,
                             absl::Span<const int64_t> batch_dims,
                             absl::Span<const int64_t> row_dims,
                             absl::Span<const int64_t> col_dims);

  Status SetDotLayout(const HloInstruction* instruction,
                      LayoutConstraints* constraints);

  bool PropagateReductionLayoutToOperand(const HloInstruction* user) override;

  se::StreamExecutor* stream_executor_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_LAYOUT_ASSIGNMENT_H_
