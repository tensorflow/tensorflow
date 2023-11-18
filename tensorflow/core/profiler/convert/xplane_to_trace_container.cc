/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/xplane_to_trace_container.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/core/profiler/convert/trace_viewer/trace_event_arguments_builder.h"
#include "tensorflow/core/profiler/convert/trace_viewer/trace_events_util.h"
#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"
#include "tensorflow/core/profiler/protobuf/trace_events_raw.pb.h"
#include "tsl/profiler/utils/tf_xplane_visitor.h"
#include "tsl/profiler/utils/timespan.h"
#include "tsl/profiler/utils/trace_utils.h"
#include "tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

using tsl::profiler::FindPlanesWithPrefix;
using tsl::profiler::FindPlaneWithName;
using tsl::profiler::HostEventType;
using tsl::profiler::StatType;
using tsl::profiler::XEventVisitor;
using tsl::profiler::XFlow;
using tsl::profiler::XLineVisitor;
using tsl::profiler::XPlaneVisitor;
using tsl::profiler::XStatVisitor;

struct SpecialArguments {
  std::optional<int64_t> group_id;
  absl::string_view step_name;
  bool is_async_event = false;
  // Both flow and async events share the flow specification.
  std::optional<XFlow> flow;
};

inline TraceEvent::FlowEntryType FlowEntryTypeFromDirection(
    XFlow::FlowDirection direction) {
  switch (direction) {
    case XFlow::kFlowUnspecified:
      return TraceEvent::FLOW_NONE;
    case XFlow::kFlowIn:
      return TraceEvent::FLOW_END;
    case XFlow::kFlowOut:
      return TraceEvent::FLOW_START;
    case XFlow::kFlowInOut:
      return TraceEvent::FLOW_MID;
  }
}

template <typename T>
void ConvertXStatToTraceEventArgument(const XStatVisitor& stat, T value,
                                      SpecialArguments& special_args,
                                      TraceEventArgumentsBuilder& args) {
  if (stat.Type() == StatType::kFlow) {
    special_args.flow = XFlow::FromStatValue(value);
  } else if (stat.Type() == StatType::kGroupId) {
    special_args.group_id = value;
  } else if (stat.Type() == StatType::kIsAsync) {
    special_args.is_async_event = true;
  } else {
    args.Append(stat.Name(), value);
  }
}

SpecialArguments ConvertXStatsToTraceEventArguments(
    const XEventVisitor& event, RawData* raw_data,
    TraceEventArguments* raw_args) {
  TraceEventArgumentsBuilder args(raw_args);
  SpecialArguments special_args;
  auto for_each_stat = [&special_args, &args](const XStatVisitor& stat) {
    if (tsl::profiler::IsInternalStat(stat.Type())) return;
    switch (stat.ValueCase()) {
      case XStat::kInt64Value:
        ConvertXStatToTraceEventArgument(stat, stat.IntValue(), special_args,
                                         args);
        break;
      case XStat::kUint64Value:
        ConvertXStatToTraceEventArgument(stat, stat.UintValue(), special_args,
                                         args);
        break;
      case XStat::kDoubleValue:
        args.Append(stat.Name(), stat.DoubleValue());
        break;
      case XStat::kStrValue:
      case XStat::kRefValue: {
        auto stat_value = stat.StrOrRefValue();
        if (stat.Type() == StatType::kStepName) {
          special_args.step_name = stat_value;
        }
        args.Append(stat.Name(), stat_value);
        break;
      }
      case XStat::kBytesValue:
        break;
      case XStat::VALUE_NOT_SET:
        break;
    }
  };
  // Ensure the metadata stats appear before the per-occurrence stats.
  event.Metadata().ForEachStat(for_each_stat);
  event.ForEachStat(for_each_stat);
  return special_args;
}

void ConvertXLineToTraceEventsContainer(uint32_t device_id,
                                        const XLineVisitor& line,
                                        TraceEventsContainer* container) {
  std::optional<uint32_t> resource_id;

  if (line.Name() != tsl::profiler::kCounterEventsLineName) {
    resource_id = line.DisplayId();
    Resource* resource = container->MutableResource(*resource_id, device_id);
    resource->set_resource_id(*resource_id);
    resource->set_name(std::string(line.DisplayName()));
    resource->set_num_events(line.NumEvents());
  }

  RawData raw_data;  // hoisted for performance
  line.ForEachEvent([device_id, resource_id, &raw_data,
                     container](const XEventVisitor& event) {
    int64_t event_type =
        event.Type().value_or(HostEventType::kUnknownHostEventType);
    if (tsl::profiler::IsInternalEvent(event_type)) return;
    TraceEventArguments* raw_args = raw_data.mutable_args();
    absl::string_view event_name;
    if (event.HasDisplayName()) {
      event_name = event.DisplayName();
      TraceEventArgumentsBuilder args(raw_args);
      constexpr size_t kMaxLongName = 10000;
      if (event.Name().size() > kMaxLongName) {
        args.Append("long_name",
                    absl::StrCat(event.Name().substr(0, kMaxLongName),
                                 "...<truncated>"));
      } else {
        args.Append("long_name", event.Name());
      }
    } else {
      event_name = event.Name();
    }
    SpecialArguments special_args =
        ConvertXStatsToTraceEventArguments(event, &raw_data, raw_args);
    if (!special_args.step_name.empty()) {
      event_name = special_args.step_name;
    }
    if (!resource_id) {
      container->AddCounterEvent(event_name, device_id, event.TimestampPs(),
                                 raw_data);
    } else if (special_args.flow) {
      tsl::profiler::Timespan span(event.TimestampPs(), event.DurationPs());
      if (special_args.is_async_event) {
        container->AddAsyncEvent(
            event_name, device_id, span, special_args.flow->Id(),
            FlowEntryTypeFromDirection(special_args.flow->Direction()),
            special_args.flow->Category(), &raw_data, special_args.group_id);
      } else {
        container->AddFlowEvent(
            event_name, *resource_id, device_id, span, special_args.flow->Id(),
            FlowEntryTypeFromDirection(special_args.flow->Direction()),
            special_args.flow->Category(), &raw_data, special_args.group_id);
      }
    } else {
      tsl::profiler::Timespan span(event.TimestampPs(), event.DurationPs());
      container->AddCompleteEvent(event_name, *resource_id, device_id, span,
                                  &raw_data, special_args.group_id);
    }
    // Cleanup hoisted structure for next event.
    if (raw_data.has_args()) raw_args->clear_arg();
  });
}

void ConvertXPlaneToTraceEventsContainer(uint64_t device_id,
                                         absl::string_view hostname,
                                         const XPlane& xplane,
                                         TraceEventsContainer* container) {
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&xplane);
  std::unique_ptr<ResourceGrouperInterface> resource_grouper =
      CreateDefaultResourceGrouper(device_id, plane.Name());

  if (plane.NumLines() == 0) return;

  for (const auto& [device_id, name] : resource_grouper->Devices()) {
    Device* device = container->MutableDevice(device_id);
    device->set_device_id(device_id);
    device->set_name(absl::StrCat(hostname, " ", name));
  }

  plane.ForEachLine([&](const XLineVisitor& line) {
    if (line.DisplayName() == tsl::profiler::kXlaAsyncOpLineName) return;
    if (line.NumEvents() == 0) return;
    // Capture a copy of XLineVisitor because it will go out of scope.
    uint32_t device_id = resource_grouper->GetDeviceId(line.DisplayId());
    ConvertXLineToTraceEventsContainer(device_id, line, container);
  });
}

}  // namespace

void ConvertXSpaceToTraceEventsContainer(absl::string_view hostname,
                                         const XSpace& space,
                                         TraceEventsContainer* container) {
  const XPlane* host_plane =
      FindPlaneWithName(space, tsl::profiler::kHostThreadsPlaneName);
  if (host_plane != nullptr) {
    ConvertXPlaneToTraceEventsContainer(tsl::profiler::kHostThreadsDeviceId,
                                        hostname, *host_plane, container);
  }

  std::vector<const XPlane*> device_planes =
      FindPlanesWithPrefix(space, tsl::profiler::kGpuPlanePrefix);

  if (device_planes.empty()) {
    device_planes = FindPlanesWithPrefix(space, tsl::profiler::kTpuPlanePrefix);
  }

  for (const XPlane* device_plane : device_planes) {
    ConvertXPlaneToTraceEventsContainer(
        tsl::profiler::kFirstDeviceId + device_plane->id(), hostname,
        *device_plane, container);
  }
  for (const XPlane* custom_plane :
       FindPlanesWithPrefix(space, tsl::profiler::kCustomPlanePrefix)) {
    ConvertXPlaneToTraceEventsContainer(
        tsl::profiler::kFirstCustomPlaneDeviceId + custom_plane->id(), hostname,
        *custom_plane, container);
  }
}

}  // namespace profiler
}  // namespace tensorflow
