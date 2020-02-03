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

#include "tensorflow/core/profiler/utils/group_events.h"

#include <stack>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

// Returns event type if it is a KernelLaunch or KernelExecute event.
absl::optional<int64> GetKernelEventType(const XPlaneVisitor& visitor,
                                         const XEvent& event) {
  bool found_correlation_id = false;
  bool found_device_id = false;
  for (const auto& stat : event.stats()) {
    if (visitor.GetStatType(stat) == StatType::kCorrelationId) {
      found_correlation_id = true;
    } else if (visitor.GetStatType(stat) == StatType::kDeviceId) {
      found_device_id = true;
    }
  }
  if (found_correlation_id) {
    return found_device_id ? HostEventType::kKernelLaunch
                           : HostEventType::kKernelExecute;
  }
  return absl::nullopt;
}

int64 GetEventType(const XPlaneVisitor& visitor, const XEvent& event) {
  if (absl::optional<int64> event_type = visitor.GetEventType(event)) {
    return *event_type;
  } else if (absl::optional<int64> kernel_event_type =
                 GetKernelEventType(visitor, event)) {
    // KernelLaunch and KernelExecute event types are not supported by
    // XPlaneVisitor and should be checked separately.
    // TODO(148346217): Make XPlaneVisitor support KernelLaunch and
    // KernelExecute event types.
    return *kernel_event_type;
  } else {
    return HostEventType::kUnknownHostEventType;
  }
}

const XStat* GetStat(const XPlaneVisitor& visitor, const XEvent& event,
                     int64 stat_type) {
  for (const auto& stat : event.stats()) {
    if (visitor.GetStatType(stat) == stat_type) {
      return &stat;
    }
  }
  return nullptr;
}

void SetGroupId(const XPlaneVisitor& visitor, int64 group_id, XEvent* event) {
  absl::optional<int64> maybe_group_id_stat_metadata_id =
      visitor.GetStatMetadataId(StatType::kGroupId);
  // TODO(jihochoi): Create stat metadata for group_id if not found.
  if (maybe_group_id_stat_metadata_id) {
    AddOrUpdateIntStat(*maybe_group_id_stat_metadata_id, group_id, event);
  }
}

}  // namespace

const XStat* EventNode::GetContextStat(int64 stat_type) const {
  if (const XStat* stat = GetStat(*visitor_, *event_, stat_type)) {
    return stat;
  } else if (parent_) {
    return parent_->GetContextStat(stat_type);
  }
  return nullptr;
}

std::string EventNode::GetGroupName() const {
  std::vector<std::string> name_parts;
  if (const XStat* graph_type_stat = GetContextStat(StatType::kGraphType)) {
    name_parts.push_back(graph_type_stat->str_value());
  }
  int64 step_num = group_id_.value_or(0);
  if (const XStat* step_num_stat = GetContextStat(StatType::kStepNum)) {
    step_num = step_num_stat->int64_value();
  }
  if (const XStat* iter_num_stat = GetContextStat(StatType::kIterNum)) {
    step_num += iter_num_stat->int64_value();
  }
  name_parts.push_back(absl::StrCat(step_num));
  return absl::StrJoin(name_parts, " ");
}

void EventNode::PropagateGroupId(int64 group_id) {
  group_id_ = group_id;
  SetGroupId(*visitor_, group_id, event_);
  for (const auto& child : children_) {
    child->PropagateGroupId(*group_id_);
  }
}

void EventNode::AddStepName(absl::string_view step_name) {
  AddOrUpdateStrStat(*visitor_->GetStatMetadataId(StatType::kStepName),
                     step_name, event_);
}

bool EventNode::IsNestedIn(EventNode* parent) {
  return parent && IsNested(GetEvent(), parent->GetEvent());
}

void ConnectIntraThread(const XPlaneVisitor& visitor, XPlane* plane,
                        EventNodeMap* event_node_map) {
  for (auto& line : *plane->mutable_lines()) {
    std::vector<EventNode*> parent_nodes;
    for (auto& event : *line.mutable_events()) {
      auto cur_node = absl::make_unique<EventNode>(&visitor, &event);
      while (!parent_nodes.empty()) {
        EventNode* parent_node = parent_nodes.back();
        if (cur_node->IsNestedIn(parent_node)) {
          parent_node->AddChild(cur_node.get());
          break;
        } else {
          parent_nodes.pop_back();
        }
      }
      parent_nodes.push_back(cur_node.get());
      (*event_node_map)[GetEventType(visitor, event)].push_back(
          std::move(cur_node));
    }
  }
}

void ConnectInterThread(
    const EventNodeMap& event_node_map,
    const std::vector<InterThreadConnectInfo>& connect_info_list) {
  for (const auto& connect_info : connect_info_list) {
    absl::flat_hash_map<std::vector<int64>, EventNode*> connect_map;
    const std::vector<int64>& stat_types = connect_info.stat_types;
    if (auto parent_event_node_list =
            gtl::FindOrNull(event_node_map, connect_info.parent_event_type)) {
      for (const auto& parent_event_node : *parent_event_node_list) {
        std::vector<int64> stats;
        for (auto stat_type : stat_types) {
          const XStat* stat = parent_event_node->GetContextStat(stat_type);
          if (!stat) break;
          stats.push_back(stat->value_case() == stat->kInt64Value
                              ? stat->int64_value()
                              : stat->uint64_value());
        }
        if (stats.size() == stat_types.size()) {
          connect_map[stats] = parent_event_node.get();
        }
      }
    }
    if (auto child_event_node_list =
            gtl::FindOrNull(event_node_map, connect_info.child_event_type)) {
      for (const auto& child_event_node : *child_event_node_list) {
        std::vector<int64> stats;
        for (auto stat_type : stat_types) {
          const XStat* stat = child_event_node->GetContextStat(stat_type);
          if (!stat) break;
          stats.push_back(stat->value_case() == stat->kInt64Value
                              ? stat->int64_value()
                              : stat->uint64_value());
        }
        if (stats.size() == stat_types.size()) {
          if (auto parent_event_node = gtl::FindPtrOrNull(connect_map, stats)) {
            parent_event_node->AddChild(child_event_node.get());
          }
        }
      }
    }
  }
}

void CreateEventGroup(const std::vector<int64 /*EventType*/>& root_event_types,
                      const EventNodeMap& event_node_map,
                      EventGroupNameMap* event_group_name_map) {
  int64 next_group_id = 0;
  for (int64 root_event_type : root_event_types) {
    if (auto root_event_node_list =
            gtl::FindOrNull(event_node_map, root_event_type)) {
      for (const auto& root_event_node : *root_event_node_list) {
        // Skip if it already belongs to a group.
        if (root_event_node->GetGroupId()) continue;
        int64 group_id = next_group_id++;
        root_event_node->PropagateGroupId(group_id);
        if (event_group_name_map) {
          (*event_group_name_map)[group_id] = root_event_node->GetGroupName();
          // Add step_name stat if it is a TraceContext event.
          // TODO(jihochoi): change event name instead.
          if (root_event_type == HostEventType::kTraceContext) {
            root_event_node->AddStepName((*event_group_name_map)[group_id]);
          }
        }
      }
    }
  }
}

void GroupEvents(const std::vector<InterThreadConnectInfo>& connect_info_list,
                 const std::vector<int64>& root_event_types, XSpace* space,
                 EventGroupNameMap* event_group_name_map) {
  EventNodeMap event_node_map;
  std::vector<XPlaneVisitor> visitors;
  visitors.reserve(space->planes_size());
  for (auto& plane : *space->mutable_planes()) {
    visitors.push_back(CreateTfXPlaneVisitor(&plane));
    ConnectIntraThread(visitors.back(), &plane, &event_node_map);
  }
  ConnectInterThread(event_node_map, connect_info_list);
  CreateEventGroup(root_event_types, event_node_map, event_group_name_map);
}

void GroupTfEvents(XSpace* space, EventGroupNameMap* event_group_name_map) {
  std::vector<InterThreadConnectInfo> connect_info_list(
      {{HostEventType::kFunctionRun,
        HostEventType::kExecutorStateProcess,
        {StatType::kStepId}},
       {HostEventType::kSessionRun,
        HostEventType::kExecutorStateProcess,
        {StatType::kStepId}},
       {HostEventType::kKernelLaunch,
        HostEventType::kKernelExecute,
        {StatType::kCorrelationId}}});
  const std::vector<int64 /*EventType*/> root_event_types(
      {HostEventType::kTraceContext, HostEventType::kFunctionRun,
       HostEventType::kSessionRun});
  GroupEvents(connect_info_list, root_event_types, space, event_group_name_map);
}

}  // namespace profiler
}  // namespace tensorflow
