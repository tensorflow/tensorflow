/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_INFERENCE_STATS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_INFERENCE_STATS_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/device_utils.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "tensorflow/core/profiler/protobuf/inference_stats.pb.h"
#include "tensorflow/core/profiler/utils/event_span.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

// Generates PerHostInferenceStats from the given trace events.
// For TPU, get time breakdown from device_traces. For GPU, get time breakdown
// from nonoverlapped_step_events.
// Get batching parameters from TFstreamz xplane in <xspace>.
void GenerateInferenceStats(
    const std::vector<tensorflow::profiler::XPlane*>& device_traces,
    const tensorflow::profiler::StepEvents& nonoverlapped_step_events,
    const tsl::profiler::GroupMetadataMap& group_metadata_map,
    const tensorflow::profiler::XSpace& xspace,
    tsl::profiler::DeviceType device_type, int32_t host_id,
    tensorflow::profiler::InferenceStats* inference_stats);

// Parses model name from TFstreamz.
// Returns whether the parsing is successful and the actual model name. If
// parsing failed, returns false and an empty string.
std::pair<bool, absl::string_view> ParseModelName(absl::string_view param);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_INFERENCE_STATS_H_
