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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_MARKER_DATA_PARSER_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_MARKER_DATA_PARSER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_activity.h"

namespace xla {
namespace profiler {

// Returns the list of activity kinds for supported marker data kinds.
// Start supporting marker data from 13.0 with CUpti_ActivityMarkerData2.
// So before that, return empty list. The list will be used when starting
// cupti activity tracing.
std::optional<CUpti_ActivityKind> GetActivityMarkerDataKind();

std::optional<std::pair<std::string, uint32_t>> ParseMarkerDataActivity(
    void* marker_data_activity);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_MARKER_DATA_PARSER_H_
