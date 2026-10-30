/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_HLO_PROFILING_UTILS_H_
#define XLA_SERVICE_HLO_PROFILING_UTILS_H_

#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"

namespace xla {

// The point in compilation at which a module's fingerprint is taken.
enum class HloModuleFingerprintStage {
  // Before the scheduler runs. Profile-guided latency estimation looks up the
  // costs of a module's instructions by this fingerprint.
  kBeforeScheduling,
};

// Computes the fingerprint of `module`, records it in the module as the
// fingerprint at `stage`, and returns it.
std::string TagWithFingerprint(HloModule* module,
                               HloModuleFingerprintStage stage);

// The fingerprint that TagWithFingerprint recorded in the module at `stage`,
// if any.
std::optional<std::string> GetHloModuleFingerprint(
    const HloModule& module, HloModuleFingerprintStage stage);
std::optional<std::string> GetHloModuleFingerprint(
    const HloModuleProto& module_proto, HloModuleFingerprintStage stage);

// The name under which a PGLE profile records the cost of the instruction
// `op_name` of the module whose fingerprint before scheduling is
// `fingerprint`, so that the costs of modules that share instruction names stay
// apart.
std::string FingerprintedCostName(absl::string_view fingerprint,
                                  absl::string_view op_name);

// The parts of a name made by FingerprintedCostName.
struct FingerprintedCostNameParts {
  std::string fingerprint;
  std::string op_name;
};

// Splits a name made by FingerprintedCostName into the fingerprint and the
// instruction name. Returns nullopt for a name that carries no fingerprint.
std::optional<FingerprintedCostNameParts> SplitFingerprintedCostName(
    absl::string_view cost_name);

}  // namespace xla

#endif  // XLA_SERVICE_HLO_PROFILING_UTILS_H_
