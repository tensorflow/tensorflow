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

#ifndef XLA_BACKENDS_CPU_TRANSFORMS_LIBRARY_MATCHER_H_
#define XLA_BACKENDS_CPU_TRANSFORMS_LIBRARY_MATCHER_H_

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla::cpu {

class LibraryMatcher {
 public:
  explicit LibraryMatcher(const TargetMachineFeatures* target_machine_features)
      : target_machine_features_(target_machine_features) {}
  virtual ~LibraryMatcher() = default;

  // Returns the set of supported HLO instructions.
  virtual absl::flat_hash_set<HloOpcode> SupportedOps() const = 0;

  // Returns true if the HLO instruction is supported by the library.
  virtual absl::StatusOr<bool> IsOpSupported(const HloInstruction* instr) {
    return false;
  }

  // Returns the output type of the library op, so we can insert a convert node
  // if the op does not support the original HLO output type.
  virtual PrimitiveType LibraryOpOutputType(const HloInstruction* instr) {
    return instr->shape().element_type();
  }

  // Returns a prefix string for the fusion op's name.
  virtual std::string fusion_prefix() const { return ""; }

  // Returns a string for FusionBackendConfig's fusion kind.
  virtual absl::string_view fusion_kind() const { return ""; }

 protected:
  const TargetMachineFeatures* target_machine_features_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_TRANSFORMS_LIBRARY_MATCHER_H_
