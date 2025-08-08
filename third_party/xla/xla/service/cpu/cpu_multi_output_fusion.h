/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_CPU_MULTI_OUTPUT_FUSION_H_
#define XLA_SERVICE_CPU_CPU_MULTI_OUTPUT_FUSION_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/multi_output_fusion.h"

namespace xla::cpu {

class CpuMultiOutputFusion final : public MultiOutputFusion {
 public:
  CpuMultiOutputFusion() = default;

  absl::string_view name() const override { return "cpu_multi_output_fusion"; }

 private:
  bool ShapesCompatibleForFusion(HloInstruction* instr1,
                                 HloInstruction* instr2) override;
  bool IsFusible(HloInstruction* instr) override;
  bool LegalToFuse(HloInstruction* instr1, HloInstruction* instr2) override;
  int64_t GetProfit(HloInstruction* instr1, HloInstruction* instr2) override;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_CPU_MULTI_OUTPUT_FUSION_H_
