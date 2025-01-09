/* Copyright 2020 The TensorFlow Authors All Rights Reserved.

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

#include "xla/tsl/profiler/convert/trace_events_to_json.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "json/json.h"
#include "xla/tsl/profiler/utils/format_utils.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/protobuf/trace_events.pb.h"

namespace tsl {
namespace profiler {
namespace {

// Converts the given time from picoseconds to microseconds and then to a string
// using maximum precision.
inline std::string PicosToMicrosString(uint64 ps) {
  return MaxPrecision(PicoToMicro(ps));
}

// Escapes and quotes the given string.
inline std::string JsonString(const std::string& s) {
  return Json::valueToQuotedString(s.c_str());
}

// Returns a vector of pointers to the elements in the given map, sorted by key.
template <typename Map>
std::vector<const typename Map::value_type*> SortByKey(const Map& m) {
  std::vector<const typename Map::value_type*> pairs;
  pairs.reserve(m.size());
  for (const auto& pair : m) {
    pairs.push_back(&pair);
  }
  absl::c_sort(pairs, [](const typename Map::value_type* a,
                         const typename Map::value_type* b) {
    return a->first < b->first;
  });
  return pairs;
}

inline void AddDeviceMetadata(uint32 device_id, const Device& device,
                              std::string* json) {
  if (!device.name().empty()) {
    absl::StrAppend(json, R"({"ph":"M","pid":)", device_id,
                    R"(,"name":"process_name","args":{"name":)",
                    JsonString(device.name()), "}},");
  }
  absl::StrAppend(json, R"({"ph":"M","pid":)", device_id,
                  R"(,"name":"process_sort_index","args":{"sort_index":)",
                  device_id, "}},");
}

inline void AddResourceMetadata(uint32 device_id, uint32 resource_id,
                                const Resource& resource, std::string* json) {
  if (!resource.name().empty()) {
    absl::StrAppend(json, R"({"ph":"M","pid":)", device_id, R"(,"tid":)",
                    resource_id, R"(,"name":"thread_name","args":{"name":)",
                    JsonString(resource.name()), "}},");
  }
  uint32 sort_index =
      resource.sort_index() ? resource.sort_index() : resource_id;
  absl::StrAppend(json, R"({"ph":"M","pid":)", device_id, R"(,"tid":)",
                  resource_id, R"(,"name":"thread_sort_index")",
                  R"(,"args":{"sort_index":)", sort_index, "}},");
}

inline void AddTraceEvent(const TraceEvent& event, string* json) {
  auto duration_ps = std::max(event.duration_ps(), protobuf_uint64{1});
  absl::StrAppend(json, R"({"ph":"X","pid":)", event.device_id(), R"(,"tid":)",
                  event.resource_id(), R"(,"ts":)",
                  PicosToMicrosString(event.timestamp_ps()), R"(,"dur":)",
                  PicosToMicrosString(duration_ps), R"(,"name":)",
                  JsonString(event.name()));
  if (!event.args().empty()) {
    absl::StrAppend(json, R"(,"args":{)");
    for (const auto* arg : SortByKey(event.args())) {
      absl::StrAppend(json, JsonString(arg->first), ":",
                      JsonString(arg->second), ",");
    }
    // Replace trailing comma with closing brace.
    json->back() = '}';
  }
  absl::StrAppend(json, "},");
}

}  // namespace

std::string TraceContainerToJson(const TraceContainer& container) {
  std::string json =
      R"({"displayTimeUnit":"ns","metadata":{"highres-ticks":true},)"
      R"("traceEvents":[)";
  for (const auto* id_and_device : SortByKey(container.trace().devices())) {
    uint32 device_id = id_and_device->first;
    const Device& device = id_and_device->second;
    AddDeviceMetadata(device_id, device, &json);
    for (const auto* id_and_resource : SortByKey(device.resources())) {
      uint32 resource_id = id_and_resource->first;
      const Resource& resource = id_and_resource->second;
      AddResourceMetadata(device_id, resource_id, resource, &json);
    }
  }
  for (const TraceEvent* const event : container.UnsortedEvents()) {
    AddTraceEvent(*event, &json);
  }
  // Add one fake event to avoid dealing with no-trailing-comma rule.
  absl::StrAppend(&json, "{}]}");
  return json;
}

}  // namespace profiler
}  // namespace tsl
