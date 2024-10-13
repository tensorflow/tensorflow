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

#include <cstdint>
#include <ostream>
#include <sstream>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_metrics_db_combiner.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/utils/event_span.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"

namespace tensorflow {
namespace profiler {

// Local core id should start from 1.
const uint32 kDefaultGpuLocalCoreId = 1;

namespace {

void StepEventsToPerCoreStepInfo(uint32_t step_num, StepDetails& step_details,
                                 PerCoreStepInfo& per_core_step_info) {
  per_core_step_info.set_step_num(step_num);
  OpMetricsDbCombiner combiner(per_core_step_info.mutable_hlo_metrics_db());
  auto step_time = step_details.StepTime();
  if (step_time.duration_ps() == 0) {
    // In case no step markers are observed for the particular step, Skip the
    // step.
    VLOG(1) << "Skipping step " << step_details.StepName()
            << "with no step markers";
    return;
  }
  for (auto& [core_id, metrics_db] : step_details.PerCoreOpMetricsDb()) {
    SetTotalTimePs(metrics_db, step_time.duration_ps());
    AddIdleOp(metrics_db);
    combiner.Combine(metrics_db);
    GenericStepBreakdown step_breakdown;
    auto& category_ps = *(step_breakdown.mutable_category_ps());
    for (auto& metric : metrics_db.metrics_db()) {
      category_ps[metric.category()] += metric.self_time_ps();
    }

    if (per_core_step_info.mutable_hlo_metrics_db()->metrics_db().empty())
      continue;
    StepInfoResult step_info;
    step_info.set_step_num(step_num);
    step_info.set_step_name(step_details.StepName());
    step_info.set_begin_ps(step_time.begin_ps());
    step_info.set_duration_ps(step_time.duration_ps());
    step_info.mutable_step_breakdown()->PackFrom(step_breakdown);
    (*per_core_step_info.mutable_step_info_per_core())[core_id] =
        std::move(step_info);
  }
}

// Converts from StepDetails to StepInfoResult.
StepInfoResult ConvertStepDetailsToStepInfo(bool has_device, int64_t step_num,
                                            StepDetails& step_details) {
  GenericStepBreakdown generic;
  tsl::profiler::Timespan step_time = step_details.StepTime();
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
  bool well_formed_step = has_device ? type_ps.contains(DEVICE_COMPUTE_16) ||
                                           type_ps.contains(DEVICE_COMPUTE_32)
                                     : type_ps.contains(HOST_COMPUTE);
  StepInfoResult step_info;
  step_info.mutable_step_breakdown()->PackFrom(generic);
  if (well_formed_step) {
    step_info.set_step_num(step_num);
    step_info.set_step_name(step_details.StepName());
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
    bool has_device, bool maybe_drop_incomplete_steps,
    StepEvents& nonoverlapped_step_events) {
  StepDatabaseResult step_db;
  // Gets sorted step numbers.
  std::vector<int64_t> step_numbers;
  step_numbers.reserve(nonoverlapped_step_events.size());
  for (const auto& step_events : nonoverlapped_step_events) {
    step_numbers.push_back(step_events.first);
  }
  absl::c_sort(step_numbers);
  for (const auto& step : step_numbers) {
    auto* step_details = gtl::FindOrNull(nonoverlapped_step_events, step);
    if (step_details == nullptr) continue;
    PerCoreStepInfo per_core_step_info;
    per_core_step_info.set_step_num(step);
    if (!step_details->PerCoreOpMetricsDb().empty()) {
      StepEventsToPerCoreStepInfo(step, *step_details, per_core_step_info);
    } else {
      StepInfoResult step_info =
          ConvertStepDetailsToStepInfo(has_device, step, *step_details);
      if (step_info.duration_ps() == 0)
        continue;  // Do not include non-well-formed steps.
      // When we generated StepEvents, we already put events from all device
      // cores and cpu threads on this host into a single event stream,
      // therefore we can't separate them anymore. Simply assigns all events to
      // Core-0.
      (*per_core_step_info
            .mutable_step_info_per_core())[kDefaultGpuLocalCoreId] =
          std::move(step_info);
      VLOG(2)
          << std::endl
          << "step_id: " << step << ", step_info:" << std::endl
          << DebugStepInfo(
                 (*per_core_step_info
                       .mutable_step_info_per_core())[kDefaultGpuLocalCoreId]);
      // Populates the collective ops information.
      auto& collectives = *per_core_step_info.mutable_all_reduce_db_per_core();
      for (const auto& it : step_details->Collectives()) {
        collectives[it.first] = it.second;
      }
      // Populates the device transfer stats for this step.
      auto& device_memory_transfers =
          *per_core_step_info.mutable_device_memory_transfers();
      for (const auto& dma : step_details->DeviceMemoryTransfers()) {
        *device_memory_transfers.Add() = dma;
      }
    }
    // The remaining fields in PerCoreStepInfo are not filled.
    *step_db.add_step_sequence() = per_core_step_info;
  }

  // If we are using sampling mode and we get enough steps, we would like to
  // drop the incomplete steps at the beginning and the end.
  // (Sometimes CUTPI instrumentation will prolong the first step too).
  int kDropIncomplteteStepThreshold = 5;
  if (maybe_drop_incomplete_steps &&
      step_db.step_sequence_size() > kDropIncomplteteStepThreshold) {
    step_db.mutable_step_sequence()->erase(
        step_db.mutable_step_sequence()->begin());
    step_db.mutable_step_sequence()->RemoveLast();
  }
  return step_db;
}

}  // namespace profiler
}  // namespace tensorflow
