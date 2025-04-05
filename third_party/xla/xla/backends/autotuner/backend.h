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

#ifndef XLA_BACKENDS_AUTOTUNER_BACKEND_H_
#define XLA_BACKENDS_AUTOTUNER_BACKEND_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "third_party/protobuf/message.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"

namespace xla {

using BackendConfig = tsl::protobuf::Message;

class Backend {
 public:
  virtual ~Backend() = default;
  explicit Backend(
      absl::string_view name,
      std::optional<Compiler::TargetConfig> target_config = std::nullopt)
      : name_(name), target_config_(target_config) {}

  absl::string_view name() const { return name_; }

  // Returns all supported configs for the given HLO instruction.
  virtual absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(HloInstruction* instr) = 0;

  // Returns a default config for the given HLO instruction.
  virtual absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      HloInstruction* instr) {
    return absl::UnimplementedError("Not implemented.");
  };

  // Wraps the HLO instruction in a module, assigns the given config, and
  // compiles it.
  virtual absl::StatusOr<std::unique_ptr<Executable>> Compile(
      const HloInstruction& hlo_instruction, const BackendConfig& config) = 0;

 protected:
  std::string name_;
  // Describes the target device, can be skipped if not needed by the backend.
  std::optional<Compiler::TargetConfig> target_config_;
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_BACKEND_H_
