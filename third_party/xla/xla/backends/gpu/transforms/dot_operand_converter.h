/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_DOT_OPERAND_CONVERTER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_DOT_OPERAND_CONVERTER_H_

#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/expanders/op_expander_pass.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"

namespace xla::gpu {

// Converts both operands of a dot to a common element type (typically the
// higher-precision operand type, or BF16 when multiplying an <=8-bit integer
// operand by an F32 operand with default precision).
class DotOperandConverter : public OpExpanderPass {
 public:
  explicit DotOperandConverter(const se::GpuComputeCapability& gpu_version =
                                   se::CudaComputeCapability::Ampere(),
                               HloPredicate extra_filter = nullptr)
      : OpExpanderPass(std::move(extra_filter)), gpu_version_(gpu_version) {}

  explicit DotOperandConverter(HloPredicate extra_filter)
      : DotOperandConverter(se::CudaComputeCapability::Ampere(),
                            std::move(extra_filter)) {}

  static bool SupportsBf16Dot(const se::GpuComputeCapability& gpu_version);

  absl::string_view name() const override { return "operand_converter"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;

 private:
  se::GpuComputeCapability gpu_version_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_DOT_OPERAND_CONVERTER_H_
