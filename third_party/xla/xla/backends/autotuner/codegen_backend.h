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

#ifndef XLA_BACKENDS_AUTOTUNER_CODEGEN_BACKEND_H_
#define XLA_BACKENDS_AUTOTUNER_CODEGEN_BACKEND_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/executable.h"
#include "tsl/platform/protobuf.h"

namespace xla {

using BackendConfig = tsl::protobuf::Message;

// Interface for a codegen backend which can compile HLO instructions with
// different configurations. This can be used to get the supported configs, and
// compile HLO instructions with different configs.
class CodegenBackend {
 public:
  virtual ~CodegenBackend() = default;

  virtual absl::string_view name() const = 0;

  // Returns all supported configs for the given HLO instruction.
  virtual std::vector<std::unique_ptr<BackendConfig>> GetSupportedConfigs(
      const HloInstruction& instr) = 0;

  // Returns a default config for the given HLO instruction.
  virtual absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) {
    return absl::UnimplementedError("Not implemented.");
  };

  // Wraps the HLO instruction in a module, assigns the given config, and
  // compiles it.
  virtual absl::StatusOr<std::unique_ptr<Executable>> Compile(
      const HloInstruction& instr, const BackendConfig& config) = 0;
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_CODEGEN_BACKEND_H_
