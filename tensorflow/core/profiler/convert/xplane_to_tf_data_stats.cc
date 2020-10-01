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

#include "tensorflow/core/profiler/convert/xplane_to_tf_data_stats.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/profiler/utils/group_events.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

// Returns true if the given iterator event is for a root iterator.
bool IsRootIteratorEvent(const XEventVisitor& iterator_event) {
  std::vector<absl::string_view> split_result =
      absl::StrSplit(iterator_event.Name(), "::");
  // The root iterator's name contains only its own name (no parent
  // information).
  return split_result.size() == 2;
}

// Returns true if the given iterator event name is for an async iterator.
bool IsAsyncIterator(absl::string_view iterator_event_name) {
  static auto* kAsyncIterators = new absl::flat_hash_set<absl::string_view>(
      {"Prefetch", "ParallelInterleave", "ParallelMap", "ParseExample",
       "MapAndBatch", "DataService", "LegacyParallelInterleave"});
  return kAsyncIterators->contains(iterator_event_name);
}

void SetIteratorMetadata(int64 id, const XEventVisitor& event,
                         IteratorMetadata* metadata) {
  metadata->set_id(id);
  auto parent_id_stat = event.GetStat(StatType::kParentId);
  if (parent_id_stat.has_value()) {
    metadata->set_parent_id(parent_id_stat->IntValue());
  }
  metadata->set_name(IteratorName(event.Name()));
  metadata->set_long_name(event.Name());
  metadata->set_is_async(IsAsyncIterator(metadata->name()));
  // TODO(b/161831651): Set params.
}

// Returns the parent iterator's id if it is a root of a device input
// pipeline.
absl::optional<int64> FindDeviceInputPipeline(const XEventVisitor& event) {
  if (event.Type() == HostEventType::kDeviceInputPipelineSecondIterator) {
    auto parent_id_stat = event.GetStat(StatType::kParentId);
    if (parent_id_stat.has_value()) return parent_id_stat->IntValue();
  }
  return absl::nullopt;
}

// Processes EventForest to do the following:
// (1) set iterator metadata
// (2) find root iterator events
// (3) find device input pipeline ids
void ProcessEventForest(const EventForest& event_forest,
                        absl::flat_hash_set<int64>* device_input_pipeline_ids,
                        absl::flat_hash_map<int64, std::vector<EventNode*>>*
                            root_iterator_event_map,
                        TfDataStats* tf_data_stats) {
  const EventNodeMap& event_node_map = event_forest.GetEventNodeMap();
  auto iterator_event_list =
      gtl::FindOrNull(event_node_map, HostEventType::kIterator);
  if (!iterator_event_list) return;
  for (const auto& iterator_event : *iterator_event_list) {
    const XEventVisitor& iterator_event_visitor =
        iterator_event->GetEventVisitor();
    auto iterator_id_stat = iterator_event_visitor.GetStat(StatType::kStepId);
    if (!iterator_id_stat.has_value()) continue;
    int64 iterator_id = iterator_id_stat->IntValue();
    auto [it, inserted] = tf_data_stats->mutable_iterator_metadata()->insert(
        protobuf::MapPair<int64, IteratorMetadata>(iterator_id,
                                                   IteratorMetadata()));
    IteratorMetadata& metadata = it->second;
    if (inserted) {
      // First time processing this iterator.
      SetIteratorMetadata(iterator_id, iterator_event_visitor, &metadata);
    }
    if (IsRootIteratorEvent(iterator_event_visitor)) {
      // Record root iterator events.
      (*root_iterator_event_map)[iterator_id].push_back(iterator_event.get());
    }
  }
  auto device_input_pipeline_second_iterator_events = gtl::FindOrNull(
      event_node_map, HostEventType::kDeviceInputPipelineSecondIterator);
  if (!device_input_pipeline_second_iterator_events) return;
  for (const auto& iterator_event :
       *device_input_pipeline_second_iterator_events) {
    const XEventVisitor& iterator_event_visitor =
        iterator_event->GetEventVisitor();
    auto iterator_id_stat = iterator_event_visitor.GetStat(StatType::kStepId);
    if (!iterator_id_stat.has_value()) continue;
    int64 iterator_id = iterator_id_stat->IntValue();
    auto [it, inserted] = tf_data_stats->mutable_iterator_metadata()->insert(
        protobuf::MapPair<int64, IteratorMetadata>(iterator_id,
                                                   IteratorMetadata()));
    IteratorMetadata& metadata = it->second;
    if (inserted) {
      // First time processing this iterator.
      SetIteratorMetadata(iterator_id, iterator_event_visitor, &metadata);
      // Find and record device input pipeline ids.
      absl::optional<int64> device_input_pipeline_id =
          FindDeviceInputPipeline(iterator_event_visitor);
      if (device_input_pipeline_id.has_value()) {
        device_input_pipeline_ids->insert(*device_input_pipeline_id);
      }
    }
  }
}

void SetInputPipelineMetadata(int64 id, uint64 name_id,
                              bool is_device_input_pipeline,
                              InputPipelineMetadata* metadata) {
  constexpr absl::string_view kHostInputPipelinePrefix = "Host:";
  constexpr absl::string_view kDeviceInputPipelinePrefix = "Device:";
  metadata->set_id(id);
  if (is_device_input_pipeline) {
    metadata->set_type(InputPipelineMetadata::DEVICE);
    metadata->set_name(absl::StrCat(kDeviceInputPipelinePrefix, name_id));
  } else {
    metadata->set_type(InputPipelineMetadata::HOST);
    metadata->set_name(absl::StrCat(kHostInputPipelinePrefix, name_id));
  }
}

void ProcessIteratorEvent(const EventNode& iterator_event,
                          InputPipelineStat* input_pipeline_stat,
                          bool is_blocking) {
  const XEventVisitor& visitor = iterator_event.GetEventVisitor();
  auto iterator_id_stat = visitor.GetStat(StatType::kStepId);
  if (!iterator_id_stat.has_value()) return;
  int64 iterator_id = iterator_id_stat->IntValue();
  auto [it, inserted] = input_pipeline_stat->mutable_iterator_stats()->insert(
      protobuf::MapPair<int64, IteratorStat>(iterator_id, IteratorStat()));
  IteratorStat& iterator_stat = it->second;
  if (inserted) {
    iterator_stat.set_id(iterator_id);
    iterator_stat.set_start_time_ps(visitor.TimestampPs());
  }
  iterator_stat.set_duration_ps(iterator_stat.duration_ps() +
                                visitor.DurationPs());
  int64 self_time_ps = visitor.DurationPs();
  tensorflow::profiler::Timespan self_time_span = visitor.GetTimespan();
  for (EventNode* child : iterator_event.GetChildren()) {
    const XEventVisitor& child_visitor = child->GetEventVisitor();
    if (ParseTfOpFullname(child_visitor.Name()).category == Category::kTfData) {
      int64 overlap_duration_ps =
          self_time_span.OverlappedDurationPs(child_visitor.GetTimespan());
      ProcessIteratorEvent(*child, input_pipeline_stat,
                           is_blocking && overlap_duration_ps);
      // Note: Assume no overlap between child events.
      self_time_ps -= overlap_duration_ps;
    }
  }
  iterator_stat.set_self_time_ps(iterator_stat.self_time_ps() + self_time_ps);
  iterator_stat.set_is_blocking(iterator_stat.is_blocking() || is_blocking);
  iterator_stat.set_num_calls(iterator_stat.num_calls() + 1);
}

void SetBottleneckIteratorId(InputPipelineStat* input_pipeline_stat) {
  int64 bottleneck_iterator_id = 0;
  int64 max_self_time = 0;
  for (const auto& [id, iterator_stat] :
       input_pipeline_stat->iterator_stats()) {
    if (iterator_stat.is_blocking() &&
        iterator_stat.self_time_ps() > max_self_time) {
      bottleneck_iterator_id = id;
      max_self_time = iterator_stat.self_time_ps();
    }
  }
  input_pipeline_stat->set_bottleneck_iterator_id(bottleneck_iterator_id);
}

void ProcessInputPipelines(
    const absl::flat_hash_set<int64>& device_input_pipeline_ids,
    absl::flat_hash_map<int64, std::vector<EventNode*>>*
        root_iterator_event_map,
    TfDataStats* tf_data_stats) {
  protobuf::Map<int64, InputPipelineStats>* input_pipelines =
      tf_data_stats->mutable_input_pipelines();
  uint64 num_host_input_pipelines = 0;
  uint64 num_device_input_pipelines = 0;
  for (auto& [root_iterator_id, root_iterator_events] :
       *root_iterator_event_map) {
    absl::c_sort(root_iterator_events,
                 [](const EventNode* lhs, const EventNode* rhs) {
                   return lhs->GetEventVisitor().DurationPs() >
                          rhs->GetEventVisitor().DurationPs();
                 });
    auto [it, inserted] =
        input_pipelines->insert(protobuf::MapPair<int64, InputPipelineStats>(
            root_iterator_id, InputPipelineStats()));
    InputPipelineStats& input_pipeline_stats = it->second;
    InputPipelineMetadata* metadata = input_pipeline_stats.mutable_metadata();
    if (inserted) {
      bool is_device_input_pipeline =
          device_input_pipeline_ids.contains(root_iterator_id);
      uint64 name_id = is_device_input_pipeline ? num_device_input_pipelines++
                                                : num_host_input_pipelines++;
      SetInputPipelineMetadata(root_iterator_id, name_id,
                               is_device_input_pipeline, metadata);
    }
    uint64 sum_latency_ps = 0;
    uint64 min_latency_ps = UINT64_MAX;
    uint64 max_latency_ps = 0;
    for (const EventNode* root_iterator_event : root_iterator_events) {
      InputPipelineStat* stat = input_pipeline_stats.add_stats();
      ProcessIteratorEvent(*root_iterator_event, stat,
                           /*is_blocking*/ true);
      SetBottleneckIteratorId(stat);
      uint64 latency_ps = root_iterator_event->GetEventVisitor().DurationPs();
      sum_latency_ps += latency_ps;
      min_latency_ps = std::min(min_latency_ps, latency_ps);
      max_latency_ps = std::max(max_latency_ps, latency_ps);
    }
    input_pipeline_stats.set_avg_latency_ps(sum_latency_ps /
                                            root_iterator_events.size());
    input_pipeline_stats.set_min_latency_ps(min_latency_ps);
    input_pipeline_stats.set_max_latency_ps(max_latency_ps);
  }
}

}  // namespace

TfDataStats ConvertXPlaneToTfDataStats(XPlane* host_plane) {
  TfDataStats tf_data_stats;
  EventForest event_forest;
  event_forest.AddPlanes(CreateTfXPlaneVisitor, {host_plane});
  event_forest.ConnectEvents();
  event_forest.ConnectTfDataEvents();
  absl::flat_hash_set<int64> device_input_pipeline_ids;
  absl::flat_hash_map<int64, std::vector<EventNode*>> root_iterator_event_map;
  ProcessEventForest(event_forest, &device_input_pipeline_ids,
                     &root_iterator_event_map, &tf_data_stats);
  ProcessInputPipelines(device_input_pipeline_ids, &root_iterator_event_map,
                        &tf_data_stats);
  return tf_data_stats;
}

}  // namespace profiler
}  // namespace tensorflow
