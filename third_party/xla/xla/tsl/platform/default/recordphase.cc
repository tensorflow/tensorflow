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

#include "xla/tsl/platform/recordphase.h"

#include <string>
#include <vector>

#include "absl/strings/string_view.h"

namespace tsl::recordphase {
void StartPhase(const absl::string_view phase_name,
                const std::vector<absl::string_view>& dependencies) {}

std::string StartPhaseUnique(
    absl::string_view phase_name,
    const std::vector<absl::string_view>& dependencies) {
  return std::string(phase_name);
}

void EndPhase(const absl::string_view phase_name) {}
}  // namespace tsl::recordphase
