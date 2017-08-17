/* Copyright 2017 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/contrib/tpu/profiler/trace_events_to_json.h"
#include "include/json/json.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {
namespace tpu {
namespace {

using ::tensorflow::strings::Appendf;
using ::tensorflow::strings::StrAppend;

constexpr double kPicosPerMicro = 1000000.0;

inline void AppendEscapedName(string *json, const string &name) {
  StrAppend(json, "\"name\":", Json::valueToQuotedString(name.c_str()).c_str());
}

// Adds resource events for a single device.
void AddResourceMetadata(uint32 device_id,
                         const std::map<uint32, const Resource *> &resources,
                         string *json) {
  for (const auto &pair : resources) {
    uint32 resource_id = pair.first;
    const Resource &resource = *pair.second;
    if (!resource.name().empty()) {
      Appendf(json,
              R"({"ph":"M","pid":%u,"tid":%u,)"
              R"("name":"thread_name","args":{)",
              device_id, resource_id);
      AppendEscapedName(json, resource.name());
      Appendf(json, "}},");
    }
    Appendf(json,
            R"({"ph":"M","pid":%u,"tid":%u,)"
            R"("name":"thread_sort_index","args":{"sort_index":%u}},)",
            device_id, resource_id, resource_id);
  }
}

void AddDeviceMetadata(const std::map<uint32, const Device *> &devices,
                       string *json) {
  for (const auto &pair : devices) {
    uint32 device_id = pair.first;
    const Device &device = *pair.second;
    if (!device.name().empty()) {
      Appendf(json,
              R"({"ph":"M","pid":%u,"name":"process_name",)"
              R"("args":{)",
              device_id);
      AppendEscapedName(json, device.name());
      StrAppend(json, "}},");
    }
    Appendf(json,
            R"({"ph":"M","pid":%u,"name":"process_sort_index",)"
            R"("args":{"sort_index":%u}},)",
            device_id, device_id);
    // Convert to a std::map so that devices are sorted by the device id.
    std::map<uint32, const Resource *> sorted_resources;
    for (const auto &pair : device.resources()) {
      sorted_resources[pair.first] = &pair.second;
    }
    AddResourceMetadata(device_id, sorted_resources, json);
  }
}

inline void AddTraceEvent(const TraceEvent &event, string *json) {
  Appendf(json, R"({"pid":%u,"tid":%u,"ts":%.5f,)", event.device_id(),
          event.resource_id(), event.timestamp_ps() / kPicosPerMicro);
  AppendEscapedName(json, event.name());
  StrAppend(json, ",");
  if (event.duration_ps() > 0) {
    Appendf(json, R"("ph":"X","dur":%.5f},)",
            event.duration_ps() / kPicosPerMicro);
  } else {
    StrAppend(json, R"("ph":"i","s":"t"},)");
  }
}

}  // namespace

string TraceEventsToJson(const Trace &trace) {
  string json;
  Appendf(&json,
          R"({"displayTimeUnit":"ns","metadata":{"highres-ticks":true},)");
  Appendf(&json,
          R"("traceEvents":[)");
  // Convert to a std::map so that devices are sorted by the device id.
  std::map<uint32, const Device *> sorted_devices;
  for (const auto &pair : trace.devices()) {
    sorted_devices[pair.first] = &pair.second;
  }
  AddDeviceMetadata(sorted_devices, &json);
  for (const TraceEvent &event : trace.trace_events()) {
    AddTraceEvent(event, &json);
  }
  // Add one fake event to avoid dealing with no-trailing-comma rule.
  Appendf(&json, R"({}]})");
  return json;
}

}  // namespace tpu
}  // namespace tensorflow
