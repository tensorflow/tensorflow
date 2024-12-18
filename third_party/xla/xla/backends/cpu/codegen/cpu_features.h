/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_CODEGEN_CPU_FEATURES_H_
#define XLA_BACKENDS_CPU_CODEGEN_CPU_FEATURES_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/cpu_info.h"

namespace xla::cpu {

// Returns the earliest CPU generation that supports the instruction set.
absl::string_view CpuTargetFromMaxFeature(tsl::port::CPUFeature max_feature);

// Converts a string representation of a CPU feature to a CPUFeature enum.
// Returns std::nullopt if the string is not a valid CPU feature.
std::optional<tsl::port::CPUFeature> CpuFeatureFromString(
    absl::string_view cpu_feature);

// Returns true if `feature` can be enabled given the maximum allowed CPU
// feature `max_feature`.
bool ShouldEnableCpuFeature(absl::string_view feature,
                            tsl::port::CPUFeature max_feature);

struct DetectedMachineAttributes {
  // The list of features available on the current machine.
  std::vector<std::string> features;
  // Number of features that are filtered out due to the `max_feature` setting.
  int32_t num_filtered_features = 0;
};

// Detects the machine attributes of the current machine.
//
// If `max_feature` is provided, the returned attributes will be filtered
// according to the maximum allowed CPU feature.
DetectedMachineAttributes DetectMachineAttributes(
    std::optional<tsl::port::CPUFeature> max_feature);

// TODO(penporn): PJRT's CPU client also calls this function. We should
// make it get the same filtered attributes according to the `max_isa` setting.
std::vector<std::string> DetectMachineAttributes()
    ABSL_DEPRECATED("Use DetectMachineAttributes defined above instead.");

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_CPU_FEATURES_H_
