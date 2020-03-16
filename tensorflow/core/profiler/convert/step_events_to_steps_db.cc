/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/profiler/convert/step_events_to_steps_db.h"

#include <sstream>

#include "google/protobuf/any.pb.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace profiler {
namespace {

// Converts from StepDetails to StepInfoResult.
StepInfoResult ConvertStepDetailsToStepInfo(bool has_device, int64 step_num,
                                            const StepDetails& step_details) {
  GenericStepBreakdown generic;
  Timespan step_time = step_details.StepTime();
  auto& type_ps = *(generic.mutable_type_ps());
  uint64 total_event_duration = 0;
  for (const auto& event : step_details.Events()) {
    // Ignore event duration outside the step marker.
    uint64 event_duration = step_time.OverlappedDurationPs(event.span);
    type_ps[event.type] += event_duration;
    total_event_duration += event_duration;
  }
  if (total_event_duration < step_time.duration_ps()) {
    // Some time in the step is not associated with any event. Classify them as
    // "unknown time".
    type_ps[UNKNOWN_TIME] += step_time.duration_ps() - total_event_duration;
  }
  // Determines if this particular step is a well-formed one.
  bool well_formed_step = has_device ? (type_ps.contains(DEVICE_COMPUTE_16) ||
                                        type_ps.contains(DEVICE_COMPUTE_32))
                                     : type_ps.contains(HOST_COMPUTE);
  StepInfoResult step_info;
  step_info.mutable_step_breakdown()->PackFrom(generic);
  if (well_formed_step) {
    step_info.set_step_num(step_num);
    step_info.set_begin_ps(step_time.begin_ps());
    step_info.set_duration_ps(step_time.duration_ps());
  } else {
    // For a non-well-formed step, sets its duration to 0 so that it will be
    // ignored by the caller of this function.
    step_info.set_duration_ps(0);
  }
  return step_info;
}

string DebugGenericStepBreakdown(const GenericStepBreakdown& generic) {
  std::ostringstream out;
  uint64 total_ps = 0;
  const auto& type_ps_map = generic.type_ps();
  for (const auto& type_ps : type_ps_map) {
    total_ps += type_ps.second;
  }
  out << "Total ps = " << total_ps << std::endl;
  for (int type = LAST_EVENT_TYPE; type >= 0; --type) {
    const auto* ps = gtl::FindOrNull(type_ps_map, type);
    if (ps == nullptr) continue;
    double percent = (*ps * 100.0) / total_ps;
    auto event_type = static_cast<EventType>(type);
    out << PrintEventType(event_type) << ": " << percent << "%"
        << ", ps = " << *ps << std::endl;
  }
  return out.str();
}

string DebugStepInfo(const StepInfoResult& step_info) {
  std::ostringstream out;
  out << "step_num=" << step_info.step_num()
      << ", duration_ps=" << step_info.duration_ps()
      << ", begin_ps=" << step_info.begin_ps() << std::endl;
  GenericStepBreakdown generic;
  if (step_info.step_breakdown().UnpackTo(&generic)) {
    out << "Generic step breakdown:" << std::endl;
    out << DebugGenericStepBreakdown(generic) << std::endl;
  } else {
    out << step_info.step_breakdown().DebugString() << std::endl;
  }
  return out.str();
}

}  // namespace

StepDatabaseResult ConvertStepEventsToStepDb(
    bool has_device, const StepEvents& nonoverlapped_step_events) {
  StepDatabaseResult step_db;
  // Gets sorted step numbers.
  std::vector<int64> step_numbers;
  step_numbers.reserve(nonoverlapped_step_events.size());
  for (const auto& step_events : nonoverlapped_step_events) {
    step_numbers.push_back(step_events.first);
  }
  absl::c_sort(step_numbers);
  for (const auto& step : step_numbers) {
    const auto* events = gtl::FindOrNull(nonoverlapped_step_events, step);
    if (events == nullptr) continue;
    StepInfoResult step_info =
        ConvertStepDetailsToStepInfo(has_device, step, *events);
    if (step_info.duration_ps() == 0)
      continue;  // Do not include non-well-formed steps.
    PerCoreStepInfo per_core_step_info;
    per_core_step_info.set_step_num(step);
    // When we generated StepEvents, we already put events from all device
    // cores and cpu threads on this host into a single event stream, therefore
    // we can't separate them anymore. Simply assigns all events to Core-0.
    (*per_core_step_info.mutable_step_info_per_core())[0] =
        std::move(step_info);
    VLOG(2) << std::endl
            << "step_id: " << step << ", step_info:" << std::endl
            << DebugStepInfo(
                   (*per_core_step_info.mutable_step_info_per_core())[0]);
    // The remaining fields in PerCoreStepInfo are not filled.
    *step_db.add_step_sequence() = per_core_step_info;
  }
  return step_db;
}

}  // namespace profiler
}  // namespace tensorflow
