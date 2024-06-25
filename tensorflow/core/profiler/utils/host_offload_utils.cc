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
#include "tensorflow/core/profiler/utils/host_offload_utils.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {

bool HostOffloadEventProcessor::IsHostOffloadOpName(
    const XEventVisitor& event) const {
  static constexpr absl::string_view keywords[] = {"copy-start",
                                                   "copy-done",
                                                   "dynamic-slice-start",
                                                   "dynamic-slice-done",
                                                   "dynamic-update-slice-start",
                                                   "dynamic-update-slice-done"};

  for (const auto& keyword : keywords) {
    // The host_memory_label_ S(5) is used by instructions to designate tensors
    // that are on the host.
    if (absl::StrContains(event.DisplayName(), keyword) &&
        absl::StrContains(event.Name(), host_memory_label_)) {
      return true;
    }
  }
  return false;
}

std::string HostOffloadEventProcessor::GetOffloadInstructionID(
    absl::string_view op_name) const {
  std::vector<std::string> op_name_vec = absl::StrSplit(op_name, '.');

  // If no dot is found, or it's at the beginning or end of the string, return
  // a 0. Hlo opnames are not expected to have a dot followed by 0.
  if (op_name_vec.size() < 2) {
    return "0";
  }
  return op_name_vec.back();
}

std::string HostOffloadEventProcessor::GetOffloadInstructionName(
    absl::string_view op_name) const {
  // TODO(b/342469268): Get the display ID and name from the HloInstruction, not
  // just the event name.
  std::string display_id = GetOffloadInstructionID(op_name);

  size_t startPos = op_name.find("-start");
  size_t donePos = op_name.find("-done");

  absl::string_view display_opname;
  if (startPos != absl::string_view::npos) {
    display_opname = op_name.substr(0, startPos);
  } else if (donePos != absl::string_view::npos) {
    display_opname = op_name.substr(0, donePos);
  } else {
    // Invalid input format: neither "-start" nor "-done" found
    LOG(WARNING) << "Invalid op name: " << op_name;
    display_opname = op_name;
  }
  return absl::StrCat("offload-", display_opname, ".", display_id);
}

void HostOffloadEventProcessor::ProcessHostOffloadOpEvent(
    const XEventVisitor& event, std::optional<int64_t> group_id) {
  std::string display_opname = GetOffloadInstructionName(event.DisplayName());

  auto [iter, inserted] = seen_events_.try_emplace(display_opname);
  std::queue<const XEventVisitor*>& events = iter->second;

  if (absl::StrContains(event.DisplayName(), "-start")) {
    // For start events, just push them into the queue.
    events.push(&event);
    return;
  } else if (absl::StrContains(event.DisplayName(), "-done")) {
    // for done events, pop the start event and create the new event.
    // Not all start events may be traced. In this case we just skip the
    // corresponding done event.
    if (events.empty()) {
      LOG(INFO) << "No corresponding start event found for "
                << event.DisplayName();
      return;
    }
    const XEventVisitor* start_event = events.front();
    events.pop();

    // At this point, we have the corresponding start and end event.
    // Create the new event.
    tsl::profiler::Timespan event_span = tsl::profiler::Timespan::FromEndPoints(
        start_event->GetTimespan().begin_ps(), event.GetTimespan().end_ps());

    XEventMetadata* host_offload_copy_metadata =
        plane_builder_->CreateEventMetadata();
    host_offload_copy_metadata->set_display_name(display_opname);
    XEventBuilder event_builder =
        host_offload_op_line_builder_->AddEvent(*host_offload_copy_metadata);
    event_builder.SetTimespan(event_span);

    // We mark the events as async so that they are displayed on new sub-lines
    // below other async events.
    const XStatMetadata& async_stat = *plane_builder_->GetOrCreateStatMetadata(
        GetStatTypeStr(StatType::kIsAsync));
    event_builder.AddStatValue(async_stat, 1);
  }
}

}  // namespace profiler
}  // namespace tensorflow
