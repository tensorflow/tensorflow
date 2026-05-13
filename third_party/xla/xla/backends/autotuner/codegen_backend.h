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

#include "google/protobuf/any.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/executable.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

using BackendConfig = google::protobuf::Any;

// Interface for a codegen backend which can compile HLO instructions with
// different configurations. This can be used to get the supported configs, and
// compile HLO instructions with different configs.
class CodegenBackend {
 public:
  virtual ~CodegenBackend() = default;

  virtual absl::string_view name() const = 0;

  virtual autotuner::Backend backend() const = 0;

  // Returns all supported configs for the given HLO instruction.
  virtual absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) = 0;

  // Returns a default config for the given HLO instruction.
  virtual absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) {
    return absl::UnimplementedError("Not implemented.");
  };

  // Wraps the HLO instruction in a module, applies the given config, and
  // compiles it.
  virtual absl::StatusOr<std::unique_ptr<Executable>> Compile(
      const HloInstruction& instr, const BackendConfig& config) = 0;

  // Apply config to the given HLO instruction.
  // This can rarely lead to the instruction being replaced by new ones in the
  // parent computation. Please check the documentation of the specific backend
  // to understand if this is the case.
  virtual absl::Status ApplyConfig(HloInstruction& instr,
                                   const BackendConfig& config) = 0;

  // Returns true if the backend can produce numerically wrong results.
  virtual bool CanProduceWrongResults() const = 0;

  // Returns true if this instruction requires sub-fusion tuning before
  // the config can be fully evaluated/applied.
  virtual bool RequiresSubFusionTuning(const HloInstruction& instr) const {
    return false;
  }

  // Generates temporary modules for sub-fusions that need to be autotuned or
  // satisfied.
  virtual absl::StatusOr<std::vector<std::unique_ptr<HloModule>>>
  GenerateSubFusions(const HloInstruction& instr, const BackendConfig& config) {
    return absl::UnimplementedError("GenerateSubFusions not implemented.");
  }

  // Stores the child configs of sub-fusions in the parent BackendConfig using
  // a map keyed by instruction fingerprint. This materializes the search tree
  // results and ensures the configurations are stored persistently.
  virtual absl::Status StoreSubFusionConfigs(
      BackendConfig& config,
      const absl::flat_hash_map<tsl::Fprint128, BackendConfig,
                                tsl::Fprint128Hasher>& sub_fusion_configs) {
    return absl::UnimplementedError("StoreSubFusionConfigs not implemented.");
  }
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_CODEGEN_BACKEND_H_
