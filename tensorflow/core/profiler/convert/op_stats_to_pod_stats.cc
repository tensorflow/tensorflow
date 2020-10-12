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

#include "tensorflow/core/profiler/convert/op_stats_to_pod_stats.h"

#include "google/protobuf/any.pb.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/utils/diagnostics.h"
#include "tensorflow/core/profiler/utils/event_span.h"
#include "tensorflow/core/profiler/utils/time_utils.h"

namespace tensorflow {
namespace profiler {

namespace {

PodStatsRecord CreatePodStatsRecord(absl::string_view host_name,
                                    const StepInfoResult& step_info) {
  PodStatsRecord record;
  GenericStepBreakdown generic;
  bool success = step_info.step_breakdown().UnpackTo(&generic);
  DCHECK(success);
  record.set_host_name(string(host_name));
  record.set_step_num(step_info.step_num());
  record.set_total_duration_us(PicosToMicros(step_info.duration_ps()));
  auto& step_breakdown_map = *record.mutable_step_breakdown_us();
  std::vector<std::pair<uint64, std::string>> metrics;
  for (const auto& entry : generic.type_ps()) {
    step_breakdown_map[entry.first] = PicosToMicros(entry.second);
    metrics.emplace_back(
        entry.second, PrintEventTypeLabel(static_cast<EventType>(entry.first)));
  }
  std::sort(metrics.begin(), metrics.end());
  record.set_bottleneck(metrics.back().second);
  return record;
}

}  // namespace

PodStatsDatabase ConvertOpStatsToPodStats(const OpStats& op_stats) {
  PodStatsDatabase pod_stats_db;
  auto add_event = [&pod_stats_db](EventType type) {
    StepBreakdownEvents* event = pod_stats_db.add_step_breakdown_events();
    event->set_id(type);
    event->set_name(PrintEventTypeLabel(type));
  };
  add_event(HOST_COMPUTE);
  add_event(HOST_COMPILE);
  add_event(HOST_TO_HOST);
  add_event(HOST_TO_DEVICE);
  add_event(HOST_PREPARE);
  add_event(DEVICE_COLLECTIVES);
  add_event(HOST_WAIT_INPUT);
  add_event(DEVICE_TO_DEVICE);
  add_event(DEVICE_TO_HOST);
  add_event(DEVICE_COMPUTE_32);
  add_event(DEVICE_COMPUTE_16);
  add_event(DEVICE_WAIT_DEVICE);
  add_event(DEVICE_WAIT_HOST);
  add_event(UNKNOWN_TIME);

  for (const auto& step_sequence : op_stats.step_db().step_sequence()) {
    int count = 0;
    for (const auto& entry : step_sequence.step_info_per_core()) {
      *pod_stats_db.add_pod_stats_record() =
          CreatePodStatsRecord(absl::StrCat(count++), entry.second);
    }
  }
  PopulateStepDiagnostics(op_stats, pod_stats_db.mutable_diagnostics());
  return pod_stats_db;
}

}  // namespace profiler
}  // namespace tensorflow
