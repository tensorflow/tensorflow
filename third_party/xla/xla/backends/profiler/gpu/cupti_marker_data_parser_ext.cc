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

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "third_party/gpus/cuda/include/nvtx3/nvToolsExtPayload.h"
#include "xla/backends/profiler/gpu/cupti_marker_data_parser.h"
#include "xla/backends/profiler/gpu/cupti_nvtx_ext_payload.h"

namespace xla {
namespace profiler {

// For cuda version after 13.0. Return CUPTI_ACTIVITY_KIND_MARKER_DATA to
// indicate cupti activities should include it when starting tracing.
std::optional<CUpti_ActivityKind> GetActivityMarkerDataKind() {
  return CUPTI_ACTIVITY_KIND_MARKER_DATA;
}

std::optional<std::pair<std::string, uint32_t>> ParseMarkerDataActivity(
    void* marker_data_activity) {
  auto* marker_data =
      static_cast<CUpti_ActivityMarkerData2*>(marker_data_activity);
  if (marker_data->payloadKind ==
      CUPTI_METRIC_VALUE_KIND_NVTX_EXTENDED_PAYLOAD) {
    uint64_t payloadAddress =
        marker_data->payload.metricValueNvtxExtendedPayload;
    auto* payload = reinterpret_cast<nvtxPayloadData_t*>(payloadAddress);

    std::string result_str;
    CuptiParseNvtxPayload(marker_data->cuptiDomainId, payload, result_str);

    // Free the payload memory allocated by CUPTI.
    if (payload != nullptr) {
      if (payload->payload != nullptr) {
        free(const_cast<void*>(payload->payload));
        payload->payload = nullptr;
      }
      free(payload);
    }
    return std::make_pair(std::move(result_str), marker_data->id);
  }
  return std::nullopt;
}

}  // namespace profiler
}  // namespace xla
