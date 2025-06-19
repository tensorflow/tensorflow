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

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/transforms/library_matcher.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla::cpu {

class OneDnnMatcher : public LibraryMatcher {
 public:
  explicit OneDnnMatcher(const TargetMachineFeatures* target_machine_features)
      : LibraryMatcher(target_machine_features) {}
  ~OneDnnMatcher() override = default;

  // Returns the set of supported HLO instructions.
  absl::flat_hash_set<HloOpcode> SupportedOps() const override {
    static const auto* kSupportedOps = new absl::flat_hash_set<HloOpcode>{};
    return *kSupportedOps;
  }

  // Returns true if the HLO instruction is supported by the library.
  absl::StatusOr<bool> IsOpSupported(const HloInstruction* instr) override {
    return false;
  }
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_TRANSFORMS_ONEDNN_MATCHER_H_
