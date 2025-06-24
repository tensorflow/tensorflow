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

#ifndef XLA_BACKENDS_CPU_TRANSFORMS_XNN_MATCHER_H_
#define XLA_BACKENDS_CPU_TRANSFORMS_XNN_MATCHER_H_

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/transforms/library_matcher.h"
#include "xla/backends/cpu/xnn_fusion.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla::cpu {

class XnnMatcher : public LibraryMatcher {
 public:
  explicit XnnMatcher(const TargetMachineFeatures* target_machine_features)
      : LibraryMatcher(target_machine_features) {}
  ~XnnMatcher() override = default;

  // Returns the set of supported HLO instructions.
  absl::flat_hash_set<HloOpcode> SupportedOps() const override {
    static const auto* kSupportedOps = []() {
      static auto* supported_ops =
          new absl::flat_hash_set<HloOpcode>{HloOpcode::kDot};
      for (const auto& op : *GetXnnUnaryOpMap()) {
        supported_ops->insert(op.first);
      }
      for (const auto& op : *GetXnnBinaryOpMap()) {
        supported_ops->insert(op.first);
      }
      return supported_ops;
    }();
    return *kSupportedOps;
  }

  // Returns true if the HLO instruction is supported by the library.
  absl::StatusOr<bool> IsOpSupported(const HloInstruction* instr) override {
    if (instr->opcode() == HloOpcode::kDot) {
      return IsDotSupportedByXnn(
          instr->dot_dimension_numbers(), instr->operand(0)->shape(),
          instr->operand(1)->shape(), instr->shape(), target_machine_features_);
    }
    if (instr->IsElementwise()) {
      return IsElementwiseOpSupportedByXnn(instr);
    }
    return false;
  }

  // Returns the output type of the XNN op, so we can insert a convert node if
  // the op does not support the original HLO output type.
  PrimitiveType LibraryOpOutputType(const HloInstruction* instr) override {
    auto out_type = instr->shape().element_type();
    if (instr->opcode() != HloOpcode::kDot) {
      return out_type;
    }
    return out_type == BF16 ? F32 : out_type;
  }

  // Returns a prefix string for the fusion op's name.
  std::string fusion_prefix() const override { return "xnn_"; }

  // Returns a string for FusionBackendConfig's fusion kind.
  absl::string_view fusion_kind() const override { return kXnnFusionKind; }
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_TRANSFORMS_XNN_MATCHER_H_
