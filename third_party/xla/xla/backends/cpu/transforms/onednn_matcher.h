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

#ifndef XLA_BACKENDS_CPU_TRANSFORMS_ONEDNN_MATCHER_H_
#define XLA_BACKENDS_CPU_TRANSFORMS_ONEDNN_MATCHER_H_

#include <algorithm>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/onednn_support.h"
#include "xla/backends/cpu/transforms/library_matcher.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "tsl/platform/protobuf.h"

namespace xla::cpu {
// TODO(intel-tf): Use oneDNN defined constant
static const int kMaxOneDnnFusionSize = 4;

class OneDnnMatcher : public LibraryMatcher {
 public:
  explicit OneDnnMatcher(const TargetMachineFeatures* target_machine_features,
                         const tsl::protobuf::RepeatedField<int>* fusion_types)
      : LibraryMatcher(target_machine_features, fusion_types) {}
  ~OneDnnMatcher() override = default;

  // Returns the set of supported HLO instructions.
  absl::flat_hash_set<HloOpcode> SupportedOps() const override {
    static const auto* kSupportedOps = []() {
      static auto* supported_ops =
          new absl::flat_hash_set<HloOpcode>{HloOpcode::kDot};
      for (const auto& [op, _] : GetOneDnnUnaryOpMap()) {
        supported_ops->insert(op);
      }
      for (const auto& [op, _] : GetOneDnnBinaryOpMap()) {
        supported_ops->insert(op);
      }
      return supported_ops;
    }();
    return *kSupportedOps;
  }

  // Returns true if the HLO instruction is supported by the library.
  absl::StatusOr<bool> IsOpSupported(const HloInstruction* instr) override {
    if (!SupportedOps().contains(instr->opcode())) {
      return false;
    }
    return IsOpSupportedByOneDnn(instr, target_machine_features_);
  }

  // Returns true if we should start a new fusion containing just the given HLO
  // instruction.
  bool ShouldCreateFusion(const HloInstruction* instr) override {
    // Policy: Only dots can start a fusion for now.
    return instr->opcode() == HloOpcode::kDot;
  }

  // Returns true if there is a limit on the number of ops in the fusion and
  // the maximum fusion size is already reached.
  bool ReachedMaxFusionSize(int fused_op_count) override {
    return fused_op_count >= kMaxOneDnnFusionSize;
  }

  // oneDNN library does not support merging fusions.
  // TODO(intel-tf): Evaluate if merging fusions has performance benefit for
  // oneDNN.
  bool ShouldMergeFusions() override { return false; }

  // Returns a prefix string for the fusion op's name.
  std::string fusion_prefix() const override { return "onednn_"; }

  // Returns a string for FusionBackendConfig's fusion kind.
  absl::string_view fusion_kind() const override { return kOneDnnFusionKind; }
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_TRANSFORMS_ONEDNN_MATCHER_H_
