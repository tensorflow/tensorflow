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

#include "xla/backends/profiler/gpu/cupti_marker_data_parser.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_activity.h"

namespace xla {
namespace profiler {

// For early version of cuda/cupti, where marker data is not fully supported,
// return nullopt for not including marker data.
std::optional<CUpti_ActivityKind> GetActivityMarkerDataKind() {
  return std::nullopt;
}

std::optional<std::pair<std::string, uint32_t>> ParseMarkerDataActivity(
    void* marker_data_activity) {
  return std::nullopt;
}

}  // namespace profiler
}  // namespace xla
