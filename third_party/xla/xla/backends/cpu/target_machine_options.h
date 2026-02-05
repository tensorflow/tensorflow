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

#ifndef XLA_BACKENDS_CPU_TARGET_MACHINE_OPTIONS_H_
#define XLA_BACKENDS_CPU_TARGET_MACHINE_OPTIONS_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/xla.pb.h"

namespace xla {
namespace cpu {

// Helper class to manage the target machine options for CPU compilation.
class TargetMachineOptions {
 public:
  // Creates a TargetMachineOptions object from the given DebugOptions. This
  // will create a TargetMachineOptions object for the host machine.
  explicit TargetMachineOptions(const DebugOptions& debug_options);

  // Creates a TargetMachineOptions object from the given triple, cpu, and
  // features.
  TargetMachineOptions(absl::string_view triple, absl::string_view cpu,
                       absl::string_view features);

  bool operator==(const TargetMachineOptions& other) const;

  TargetMachineOptionsProto ToProto() const;
  static absl::StatusOr<TargetMachineOptions> FromProto(
      const TargetMachineOptionsProto& proto);

  const std::string& triple() const { return triple_; }
  const std::string& cpu() const { return cpu_; }
  const std::vector<std::string>& enabled_features() const {
    return enabled_features_;
  }
  const std::vector<std::string>& disabled_features() const {
    return disabled_features_;
  }

  absl::Status SetFeatures(absl::string_view features);

  // Returns the target machine features in the format that LLVM understands
  // (e.x. "+avx2,-avx512")).
  std::string GetTargetMachineFeatures() const;

  // Returns the target machine features in the format that LLVM understands -
  // features prefixed with "+" or "-". E.x. {"+avx2", "-avx512"}.
  std::vector<std::string> GetTargetMachineFeaturesVector() const;

 private:
  TargetMachineOptions() = default;

  std::string triple_;
  std::string cpu_;
  std::vector<std::string> enabled_features_;
  std::vector<std::string> disabled_features_;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_BACKENDS_CPU_TARGET_MACHINE_OPTIONS_H_
