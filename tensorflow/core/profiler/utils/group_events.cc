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

#include <algorithm>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

// Creates stat metadata for the stats which may be added by grouping.
void CreateStatMetadata(XPlane* plane) {
  XPlaneBuilder builder(plane);
  builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId));
  builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kStepName));
  builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kIsEager));
  builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSelectedGroupIds));
}

// Returns event type if it is a KernelLaunch or KernelExecute event.
absl::optional<int64> GetKernelEventType(bool is_host_plane,
                                         const EventNode& event) {
  if (event.GetEventVisitor().GetStat(StatType::kCorrelationId).has_value()) {
    return is_host_plane ? HostEventType::kKernelLaunch
                         : HostEventType::kKernelExecute;
  }
  return absl::nullopt;
}

int64 GetEventType(bool is_host_plane, const EventNode& event) {
  if (absl::optional<int64> event_type = event.GetEventVisitor().Type()) {
    return *event_type;
  } else if (absl::optional<int64> kernel_event_type =
                 GetKernelEventType(is_host_plane, event)) {
    // KernelLaunch and KernelExecute event types are not supported by
    // XPlaneVisitor and should be checked separately.
    // TODO(b/148346217): Make XPlaneVisitor support KernelLaunch and
    // KernelExecute event types.
    return *kernel_event_type;
  } else {
    return HostEventType::kUnknownHostEventType;
  }
}

void SetContextGroup(EventNode* event, ContextGroupMap* context_groups) {
  auto producer = event->GetProducerContext();
  if (producer.has_value()) {
    ((*context_groups)[producer->type][producer->id])
        .producers.push_back(event);
  }
  auto consumer = event->GetConsumerContext();
  if (consumer.has_value()) {
    ((*context_groups)[consumer->type][consumer->id])
        .consumers.push_back(event);
  }
}

void ConnectContextGroups(const ContextGroupMap& context_groups) {
  for (auto& type_id_group : context_groups) {
    for (auto& id_group : type_id_group.second) {
      const ContextGroup& group = id_group.second;
      for (EventNode* parent : group.producers) {
        for (EventNode* child : group.consumers) {
          parent->AddChild(child);
        }
      }
    }
  }
}

std::unique_ptr<XEvent> CreateVirtualEvent(const XStat& step_id_stat,
                                           const XStat& iter_num_stat) {
  auto virtual_event = absl::make_unique<XEvent>();
  *virtual_event->add_stats() = step_id_stat;
  *virtual_event->add_stats() = iter_num_stat;
  return virtual_event;
}

bool HasFunctionRun(EventNode* event_node) {
  for (EventNode* child : event_node->GetChildren()) {
    if (child->GetEventVisitor().Type() == HostEventType::kFunctionRun) {
      return true;
    }
  }
  return false;
}

bool IsImplicitRootEvent(const XEventVisitor& event) {
  static const auto* const kImplicitRootEvents = new absl::flat_hash_set<int64>{
      HostEventType::kFunctionRun, HostEventType::kSessionRun,
      HostEventType::kRunGraph, HostEventType::kExecutorStateProcess};
  return event.Type().has_value() &&
         kImplicitRootEvents->contains(*event.Type());
}

void ProcessRootEvent(int64 group_id, bool set_step_name, EventNode* root_event,
                      GroupMetadataMap* group_metadata_map) {
  root_event->PropagateGroupId(group_id, group_metadata_map);
  if (!set_step_name) {
    // Step names are not necessary for inference profiles but add group_id to
    // group_metadata_map to count the number of groups.
    group_metadata_map->emplace(group_id, GroupMetadata());
    return;
  }
  std::string group_name = root_event->GetGroupName();
  // TODO(jihochoi): change event name instead.
  if (!IsImplicitRootEvent(root_event->GetEventVisitor())) {
    // Add the `step_name` stat for the user-defined root events only. When an
    // XEvent is converted to a trace event, the trace event name is set to the
    // `step_name` stat's value if present.
    root_event->AddStepName(group_name);
  }
  (*group_metadata_map)[group_id].name = std::move(group_name);
}

bool IsTfDataEvent(const EventNode& event_node) {
  return event_node.FindParent(HostEventType::kTfDataCapturedFunctionRun) ||
         event_node.FindParent(
             HostEventType::kTfDataCapturedFunctionRunAsync) ||
         event_node.FindParent(
             HostEventType::kTfDataCapturedFunctionRunInstantiated) ||
         event_node.FindParent(
             HostEventType::kTfDataCapturedFunctionRunWithBorrowedArgs);
}

struct ContextTypeAndId {
  int type;
  uint64 id;
};

absl::optional<ContextTypeAndId> GetLegacyProducerContext(
    const XEventVisitor& event) {
  absl::optional<ContextTypeAndId> type_and_id;
  absl::optional<int64> event_type = event.Type();
  if (event_type.has_value()) {
    switch (*event_type) {
      case HostEventType::kTraceContext:
      case HostEventType::kFunctionRun:
      case HostEventType::kSessionRun:
      case HostEventType::kRunGraph: {
        absl::optional<XStatVisitor> stat = event.GetStat(StatType::kStepId);
        if (stat.has_value()) {
          type_and_id = {static_cast<int>(ContextType::kTfExecutor),
                         static_cast<uint64>(stat->IntValue())};
        }
        break;
      }
      case HostEventType::kCallOp:
      case HostEventType::kNumericalGradientOpEvalRight:
      case HostEventType::kNumericalGradientOpEvalLeft:
      case HostEventType::kSymbolicGradientOp:
      case HostEventType::kRemoteCallOp:
      case HostEventType::kIfOp:
      case HostEventType::kCaseOp:
      case HostEventType::kPartitionedCallOp: {
        // TODO(b/154510598): Fix handling of the loop ops.
        // case HostEventType::kWhileOpEvalCond:
        // case HostEventType::kWhileOpStartBody:
        // case HostEventType::kForOp:
        // case HostEventType::kParallelForOp:
        // case HostEventType::kForeverOp:
        absl::optional<XStatVisitor> stat =
            event.GetStat(StatType::kFunctionStepId);
        if (stat.has_value()) {
          type_and_id = {static_cast<int>(ContextType::kTfExecutor),
                         static_cast<uint64>(stat->IntValue())};
        }
        break;
      }
      default:
        break;
    }
  }
  return type_and_id;
}

absl::optional<ContextTypeAndId> GetLegacyConsumerContext(
    const XEventVisitor& event) {
  absl::optional<ContextTypeAndId> type_and_id;
  absl::optional<int64> event_type = event.Type();
  if (event_type.has_value()) {
    switch (*event_type) {
      case HostEventType::kExecutorStateProcess:
      case HostEventType::kExecutorDoneCallback:
      case HostEventType::kRunGraphDone: {
        absl::optional<XStatVisitor> stat = event.GetStat(StatType::kStepId);
        if (stat.has_value()) {
          type_and_id = {static_cast<int>(ContextType::kTfExecutor),
                         static_cast<uint64>(stat->IntValue())};
        }
        break;
      }
      default:
        break;
    }
  }
  return type_and_id;
}

bool IsLegacyRootEvent(const XEventVisitor& event) {
  static const auto* const kRootEvents = new absl::flat_hash_set<int64>{
      HostEventType::kTraceContext, HostEventType::kFunctionRun,
      HostEventType::kSessionRun, HostEventType::kRunGraph};
  return event.Type().has_value() && kRootEvents->contains(*event.Type());
}

using Comparator = std::function<bool(const EventNode*)>;

const EventNode* FindParentWithComparator(const Comparator& comparator,
                                          const EventNode* node,
                                          bool include_self) {
  std::queue<const EventNode*> nodes;
  absl::flat_hash_set<const EventNode*> seen = {node};
  if (include_self) {
    nodes.push(node);
  } else {
    for (const EventNode* parent : node->GetParents()) {
      nodes.push(parent);
      seen.insert(parent);
    }
  }
  while (!nodes.empty()) {
    const EventNode* node = nodes.front();
    nodes.pop();
    if (comparator(node)) return node;
    for (const EventNode* parent : node->GetParents()) {
      if (seen.contains(parent)) continue;
      nodes.push(parent);
      seen.insert(parent);
    }
  }
  return nullptr;
}

// Returns true if none of its ancestors is a root event.
bool IsTopRoot(const EventNode* event) {
  // If it is already grouped, it is not a top root.
  if (event->GetGroupId().has_value()) return false;
  const EventNode* root_parent = FindParentWithComparator(
      [](const EventNode* node) { return node->IsRoot(); }, event,
      /*include_self=*/false);
  return root_parent == nullptr;
}

void SortEventList(EventList* event_list) {
  absl::c_sort(*event_list, [](const EventNode* e1, const EventNode* e2) {
    return e1->GetEventVisitor().TimestampPs() <
           e2->GetEventVisitor().TimestampPs();
  });
}

// Returns true if it has JAX-related events.
bool HasJaxEvent(const EventNodeMap& event_node_map) {
  return event_node_map.contains(HostEventType::kExecuteOnLocalDevices);
}

bool IsIteratorEventType(absl::optional<int64> event_type) {
  return event_type == HostEventType::kIterator ||
         event_type == HostEventType::kDeviceInputPipelineSecondIterator;
}

}  // namespace

// Returns true if TF's loop ops exist in the given XSpace's metadata.
bool CheckLoopOp(const XSpace& space) {
  for (const XPlane& plane : space.planes()) {
    for (const auto& event_metadata : plane.event_metadata()) {
      absl::optional<int64> event_type =
          FindHostEventType(event_metadata.second.name());
      if (!event_type.has_value()) continue;
      switch (*event_type) {
        case HostEventType::kWhileOpEvalCond:
        case HostEventType::kWhileOpStartBody:
        case HostEventType::kForOp:
        case HostEventType::kParallelForOp:
        case HostEventType::kForeverOp:
          return true;
        default:
          break;
      }
    }
  }
  return false;
}

EventNode::EventNode(const XPlaneVisitor* plane, XLine* raw_line,
                     XEvent* raw_event)
    : plane_(plane),
      visitor_(plane, raw_line, raw_event),
      raw_line_(raw_line),
      raw_event_(raw_event) {
  absl::optional<int> producer_type;
  absl::optional<uint64> producer_id;
  absl::optional<int> consumer_type;
  absl::optional<uint64> consumer_id;

  visitor_.ForEachStat([&](const XStatVisitor& stat) {
    if (!stat.Type().has_value()) return;
    switch (*stat.Type()) {
      case StatType::kProducerType:
        producer_type = stat.IntValue();
        break;
      case StatType::kProducerId:
        producer_id = stat.IntOrUintValue();
        break;
      case StatType::kConsumerType:
        consumer_type = stat.IntValue();
        break;
      case StatType::kConsumerId:
        consumer_id = stat.IntOrUintValue();
        break;
      case StatType::kIsRoot:
        is_root_ = stat.IntValue();
        break;
      case StatType::kIsAsync:
        is_async_ = stat.IntValue();
        break;
      default:
        break;
    }
  });

  // Support legacy traces.
  if (!producer_type.has_value() || !producer_id.has_value()) {
    if (auto producer_context = GetLegacyProducerContext(visitor_)) {
      producer_type = producer_context->type;
      producer_id = producer_context->id;
    }
  }
  if (!consumer_type.has_value() || !consumer_id.has_value()) {
    if (auto consumer_context = GetLegacyConsumerContext(visitor_)) {
      consumer_type = consumer_context->type;
      consumer_id = consumer_context->id;
    }
  }
  is_root_ = is_root_ || IsLegacyRootEvent(visitor_);

  if (producer_type.has_value() && producer_id.has_value()) {
    producer_context_ = {*producer_type, *producer_id};
  }
  if (consumer_type.has_value() && consumer_id.has_value()) {
    consumer_context_ = {*consumer_type, *consumer_id};
  }
}

EventNode::EventNode(const EventNode& event_node)
    : EventNode(event_node.plane_, event_node.raw_line_,
                event_node.raw_event_) {}

absl::optional<XStatVisitor> EventNode::GetContextStat(int64 stat_type) const {
  std::queue<const EventNode*> nodes;
  absl::flat_hash_set<const EventNode*> seen = {this};
  nodes.push(this);
  while (!nodes.empty()) {
    const EventNode* node = nodes.front();
    nodes.pop();
    if (absl::optional<XStatVisitor> stat = node->visitor_.GetStat(stat_type)) {
      return stat;
    }
    for (const EventNode* parent : node->GetParents()) {
      if (seen.contains(parent)) continue;
      nodes.push(parent);
      seen.insert(parent);
    }
  }
  return absl::nullopt;
}

std::string EventNode::GetGroupName() const {
  std::string name;
  if (absl::optional<XStatVisitor> stat =
          GetContextStat(StatType::kGraphType)) {
    absl::StrAppend(&name, stat->StrOrRefValue(), " ");
  } else if (!(IsImplicitRootEvent(visitor_))) {
    absl::StrAppend(&name, GetEventVisitor().Name(), " ");
  }
  int64 step_num = group_id_.value_or(0);
  if (absl::optional<XStatVisitor> stat = GetContextStat(StatType::kIterNum)) {
    step_num = stat->IntValue();
  } else if (absl::optional<XStatVisitor> stat =
                 GetContextStat(StatType::kStepNum)) {
    step_num = stat->IntValue();
  }
  absl::StrAppend(&name, step_num);
  return name;
}

XStat* EventNode::FindOrAddStatByType(int64 stat_type) {
  const XStatMetadata* stat_metadata = plane_->GetStatMetadataByType(stat_type);
  DCHECK(stat_metadata != nullptr);
  return FindOrAddMutableStat(*stat_metadata, raw_event_);
}

void EventNode::SetGroupId(int64 group_id) {
  group_id_ = group_id;
  FindOrAddStatByType(StatType::kGroupId)->set_int64_value(group_id);
}

void EventNode::PropagateGroupId(int64 group_id,
                                 GroupMetadataMap* group_metadata_map) {
  std::queue<EventNode*> nodes;
  absl::flat_hash_set<EventNode*> seen = {this};
  nodes.push(this);
  while (!nodes.empty()) {
    EventNode* node = nodes.front();
    nodes.pop();
    absl::optional<int64> node_group_id = node->GetGroupId();
    if (node_group_id.has_value()) {
      if (*node_group_id != group_id) {
        (*group_metadata_map)[group_id].children.insert(*node_group_id);
        (*group_metadata_map)[*node_group_id].parents.insert(group_id);
      }
    } else {
      node->SetGroupId(group_id);
      for (EventNode* child : node->GetChildren()) {
        if (seen.contains(child)) continue;
        nodes.push(child);
        seen.insert(child);
      }
    }
  }
}

void EventNode::AddStepName(absl::string_view step_name) {
  FindOrAddStatByType(StatType::kStepName)
      ->set_str_value(step_name.data(), step_name.size());
}

void EventNode::AddSelectedGroupIds(
    const GroupMetadataMap& group_metadata_map) {
  const auto& group_metadata = group_metadata_map.at(*group_id_);
  std::vector<int64> group_ids;
  group_ids.reserve(1 + group_metadata.parents.size() +
                    group_metadata.children.size());
  group_ids.push_back(*group_id_);
  group_ids.insert(group_ids.end(), group_metadata.parents.begin(),
                   group_metadata.parents.end());
  group_ids.insert(group_ids.end(), group_metadata.children.begin(),
                   group_metadata.children.end());
  FindOrAddStatByType(StatType::kSelectedGroupIds)
      ->set_str_value(
          absl::StrCat("?selected_group_ids=", absl::StrJoin(group_ids, ",")));
}

void EventNode::SetIsEager(bool is_eager) {
  FindOrAddStatByType(StatType::kIsEager)->set_int64_value(is_eager ? 1 : 0);
}

bool EventNode::IsEager() {
  // It is eagerly executed if its trace context includes the EagerKernelExecute
  // event (which may execute an op eagerly or through the TF executor) but not
  // the TF executor event.
  return FindParent(HostEventType::kExecutorStateProcess) == nullptr &&
         FindParent(HostEventType::kEagerKernelExecute) != nullptr;
}

const EventNode* EventNode::FindParent(int64 event_type) const {
  return FindParentWithComparator(
      [event_type](const EventNode* node) {
        return node->GetEventVisitor().Type() == event_type;
      },
      this, /*include_self=*/true);
}

bool EventNode::StartsBefore(const EventNode& other) const {
  return GetEventVisitor().TimestampPs() <=
         other.GetEventVisitor().TimestampPs();
}

void EventForest::ConnectIntraThread(XPlane* plane, XPlaneVisitor* visitor,
                                     ContextGroupMap* context_groups) {
  // TODO(b/149095099): avoid string comparison.
  bool is_host_plane = (visitor->Name() == kHostThreadsPlaneName);
  for (auto& line : *plane->mutable_lines()) {
    std::vector<EventNode*> parent_nodes;
    for (auto& event : *line.mutable_events()) {
      auto cur_node = absl::make_unique<EventNode>(visitor, &line, &event);
      // Update `context_groups` for `ConnectInterThread`.
      SetContextGroup(cur_node.get(), context_groups);
      // Update `root_events_` for `CreateEventGroup`.
      if (cur_node->IsRoot()) root_events_.push_back(cur_node.get());
      // Async events are ignored when processing the nesting relationship.
      if (cur_node->IsAsync()) continue;
      while (!parent_nodes.empty()) {
        EventNode* parent_node = parent_nodes.back();
        if (parent_node->GetEventVisitor().GetTimespan().Includes(
                cur_node->GetEventVisitor().GetTimespan())) {
          parent_node->AddChild(cur_node.get());
          break;
        } else {
          parent_nodes.pop_back();
        }
      }
      parent_nodes.push_back(cur_node.get());
      // event_node_map_ keeps cur_node alive.
      event_node_map_[GetEventType(is_host_plane, *cur_node)].push_back(
          std::move(cur_node));
    }
  }
}

void EventForest::ConnectInterThread(
    const std::vector<InterThreadConnectInfo>& connect_info_list) {
  for (const auto& connect_info : connect_info_list) {
    absl::flat_hash_map<std::vector<uint64>, EventNode*> connect_map;
    const std::vector<int64>& parent_stat_types =
        connect_info.parent_stat_types;
    const std::vector<int64>* child_stat_types = &connect_info.child_stat_types;
    if (child_stat_types->empty()) {
      child_stat_types = &parent_stat_types;
    }
    if (auto parent_event_node_list =
            gtl::FindOrNull(event_node_map_, connect_info.parent_event_type)) {
      for (const auto& parent_event_node : *parent_event_node_list) {
        std::vector<uint64> stats;
        for (auto stat_type : parent_stat_types) {
          absl::optional<XStatVisitor> stat =
              parent_event_node->GetContextStat(stat_type);
          if (!stat) break;
          stats.push_back(stat->IntOrUintValue());
        }
        if (stats.size() == parent_stat_types.size()) {
          connect_map[stats] = parent_event_node.get();
        }
      }
    }
    if (auto child_event_node_list =
            gtl::FindOrNull(event_node_map_, connect_info.child_event_type)) {
      for (const auto& child_event_node : *child_event_node_list) {
        std::vector<uint64> stats;
        for (auto stat_type : *child_stat_types) {
          absl::optional<XStatVisitor> stat =
              child_event_node->GetContextStat(stat_type);
          if (!stat) break;
          stats.push_back(stat->IntOrUintValue());
        }
        if (stats.size() == child_stat_types->size()) {
          if (auto parent_event_node = gtl::FindPtrOrNull(connect_map, stats)) {
            parent_event_node->AddChild(child_event_node.get());
          }
        }
      }
    }
  }
}

void EventForest::ProcessUserDefinedRootEvents(
    const std::vector<int64 /*EventType*/>& user_defined_root_event_types) {
  for (int64 user_defined_root_event_type : user_defined_root_event_types) {
    if (auto root_events =
            gtl::FindOrNull(event_node_map_, user_defined_root_event_type)) {
      for (const auto& root_event : *root_events) {
        root_event->SetIsRoot(true);
        root_events_.push_back(root_event.get());
      }
    }
  }
}

void EventForest::CreateEventGroups() {
  // Handle inference batching profiles.
  if (event_node_map_.contains(HostEventType::kProcessBatch)) {
    // Assign group_id per batch.
    for (const auto& process_batch_node :
         event_node_map_[HostEventType::kProcessBatch]) {
      ProcessRootEvent(next_group_id_++, /*set_step_name=*/false,
                       process_batch_node.get(), &group_metadata_map_);
    }
    HostEventType request_event_type =
        event_node_map_.contains(HostEventType::kBatchingSessionRun)
            ? HostEventType::kBatchingSessionRun
            : HostEventType::kSessionRun;
    if (auto request_events =
            gtl::FindOrNull(event_node_map_, request_event_type)) {
      // Assign group_id per request.
      for (const auto& request_event : *request_events) {
        ProcessRootEvent(next_group_id_++, /*set_step_name=*/false,
                         request_event.get(), &group_metadata_map_);
        // Also, set a helper stat for selected_group_ids.
        request_event->AddSelectedGroupIds(group_metadata_map_);
      }
    }
    // Set a helper stat for selected_group_ids per batch.
    for (const auto& process_batch_node :
         event_node_map_[HostEventType::kProcessBatch]) {
      process_batch_node->AddSelectedGroupIds(group_metadata_map_);
    }
    return;
  }
  // Create a group for each TF loop iteration in non-JAX profiles.
  if (!HasJaxEvent(event_node_map_) && !tf_loop_root_events_.empty()) {
    for (EventNode* root_event : tf_loop_root_events_) {
      ProcessRootEvent(next_group_id_++, /*set_step_name=*/true, root_event,
                       &group_metadata_map_);
    }
    return;
  }
  SortEventList(&root_events_);
  // Create a group for each top root event while ignoring TF's legacy root
  // events for JAX profiles.
  for (EventNode* root_event : root_events_) {
    if (IsTopRoot(root_event) &&
        (!HasJaxEvent(event_node_map_) ||
         !IsLegacyRootEvent(root_event->GetEventVisitor()))) {
      ProcessRootEvent(next_group_id_++, /*set_step_name=*/true, root_event,
                       &group_metadata_map_);
    }
  }
}

void EventForest::MarkEagerlyExecutedGpuKernels() {
  auto kernel_execute_event_node_list =
      gtl::FindOrNull(event_node_map_, HostEventType::kKernelExecute);
  if (!kernel_execute_event_node_list) return;
  for (auto& kernel_execute_event_node : *kernel_execute_event_node_list) {
    kernel_execute_event_node->SetIsEager(kernel_execute_event_node->IsEager());
  }
}

void EventForest::MarkEagerlyExecutedCpuTfOps() {
  auto tf_op_run_event_node_list =
      gtl::FindOrNull(event_node_map_, HostEventType::kTfOpRun);
  if (!tf_op_run_event_node_list) return;
  for (auto& tf_op_run_event_node : *tf_op_run_event_node_list) {
    tf_op_run_event_node->SetIsEager(tf_op_run_event_node->IsEager());
  }
}

void EventForest::ProcessTensorFlowLoop() {
  struct TensorFlowLoopIteration {
    EventNode* first_event = nullptr;
    std::vector<EventNode*> events;
  };
  using TensorFlowLoop = std::map<int64 /*iter_num*/, TensorFlowLoopIteration>;
  absl::flat_hash_map<int64 /*step_id*/, TensorFlowLoop> tf_loops;

  // Sort the TF executor events by TF function/session (step_id) and iter_num.
  auto executor_event_list =
      gtl::FindOrNull(event_node_map_, HostEventType::kExecutorStateProcess);
  if (!executor_event_list) return;
  for (auto& executor_event : *executor_event_list) {
    if (IsTfDataEvent(*executor_event)) continue;
    absl::optional<XStatVisitor> step_id_stat =
        executor_event->GetContextStat(StatType::kStepId);
    absl::optional<XStatVisitor> iter_num_stat =
        executor_event->GetContextStat(StatType::kIterNum);
    if (!step_id_stat || !iter_num_stat) continue;
    int64 step_id = step_id_stat->IntValue();
    TensorFlowLoop& tf_loop = tf_loops[step_id];
    TensorFlowLoopIteration& iteration = tf_loop[iter_num_stat->IntValue()];
    if (!iteration.first_event ||
        executor_event->StartsBefore(*iteration.first_event)) {
      iteration.first_event = executor_event.get();
    }
    iteration.events.push_back(executor_event.get());
  }

  // Sort the TF loops by start time.
  std::map<int64 /*start_time*/, int64 /*step_id*/> sorted_tf_loops;
  for (const auto& step_id_and_tf_loop : tf_loops) {
    auto& iterations = step_id_and_tf_loop.second;
    // Filter out TF function/session without loops.
    if (iterations.size() == 1 && iterations.count(0)) continue;
    int64 start_time = iterations.cbegin()
                           ->second.first_event->GetEventVisitor()
                           .TimestampPs();
    DCHECK_EQ(sorted_tf_loops.count(start_time), 0);
    sorted_tf_loops[start_time] = step_id_and_tf_loop.first;
  }

  // Register the first event of each iteration as a root event. Also, add the
  // other events of the iteration as child to the root event.
  bool next_group_id_updated = false;
  for (const auto& start_time_and_step_id : sorted_tf_loops) {
    TensorFlowLoop& tf_loop = tf_loops[start_time_and_step_id.second];
    for (auto& iter_num_and_iteration : tf_loop) {
      if (!next_group_id_updated) {
        // Set next_group_id_ to the first iter_num of the first TF loop. This
        // is necessary later when selecting the intersection of the steps from
        // multiple hosts.
        next_group_id_ = iter_num_and_iteration.first;
        next_group_id_updated = true;
      }
      TensorFlowLoopIteration& iteration = iter_num_and_iteration.second;
      EventNode* root_event = iteration.first_event;
      tf_loop_root_events_.push_back(root_event);
      for (EventNode* event : iteration.events) {
        if (event == root_event) continue;
        root_event->AddChild(event);
      }
    }
  }
}

void EventForest::ProcessWorker() {
  auto eager_kernel_execute_event_list =
      gtl::FindOrNull(event_node_map_, HostEventType::kEagerKernelExecute);
  if (!eager_kernel_execute_event_list) return;
  // The last EagerKernelExecute with a FunctionRun child.
  EventNode* root_event = nullptr;
  for (auto& eager_kernel_execute_event : *eager_kernel_execute_event_list) {
    if (HasFunctionRun(eager_kernel_execute_event.get())) {
      // A function op becomes a new root.
      root_event = eager_kernel_execute_event.get();
      root_event->SetIsRoot(true);
      root_events_.push_back(root_event);
    } else if (root_event) {
      // Add non-function eager ops as child.
      root_event->AddChild(eager_kernel_execute_event.get());
    }
  }
}

void EventForest::ProcessModelIds() {
  auto session_run_event_list =
      gtl::FindOrNull(event_node_map_, HostEventType::kSessionRun);
  if (!session_run_event_list) return;
  for (const auto& session_run_event : *session_run_event_list) {
    auto group_id = session_run_event->GetGroupId();
    if (!group_id.has_value()) continue;
    absl::optional<XStatVisitor> model_id =
        session_run_event->GetEventVisitor().GetStat(StatType::kModelId);
    if (!model_id.has_value()) continue;
    group_metadata_map_[*group_id].model_id = model_id->ToString();
  }
}

void EventForest::AddPlane(
    const std::function<XPlaneVisitor(const XPlane*)> visitor_factory,
    XPlane* plane) {
  CreateStatMetadata(plane);
  planes_.push_back({plane, visitor_factory(plane)});
}

void EventForest::AddSpace(
    const std::function<XPlaneVisitor(const XPlane*)> visitor_factory,
    XSpace* space) {
  for (XPlane& plane : *space->mutable_planes()) {
    AddPlane(visitor_factory, &plane);
  }
}

void EventForest::AddPlanes(
    const std::function<XPlaneVisitor(const XPlane*)> visitor_factory,
    const std::vector<XPlane*>& planes) {
  for (XPlane* plane : planes) {
    AddPlane(visitor_factory, plane);
  }
}

void EventForest::ConnectEvents(
    const std::vector<InterThreadConnectInfo>& connect_info_list) {
  ContextGroupMap context_groups;
  for (auto& plane_visitor : planes_) {
    ConnectIntraThread(plane_visitor.first, &plane_visitor.second,
                       &context_groups);
  }
  ConnectInterThread(connect_info_list);
  ConnectContextGroups(context_groups);
}

void EventForest::ConnectTfDataEvents() {
  absl::flat_hash_map<std::pair<int64 /*iterator_id*/, int64 /*element_id*/>,
                      std::vector<EventNode*>>
      produce_iterator_map;
  uint64 num_producers = 0;
  for (HostEventType event_type :
       {HostEventType::kPrefetchProduce,
        HostEventType::kParallelInterleaveProduce,
        HostEventType::kParallelMapProduce, HostEventType::kMapAndBatchProduce,
        HostEventType::kParseExampleProduce}) {
    auto produce_event_list = gtl::FindOrNull(event_node_map_, event_type);
    if (!produce_event_list) continue;
    VLOG(1) << produce_event_list->size() << " "
            << GetHostEventTypeStr(event_type) << " events found.";
    for (auto& produce_event : *produce_event_list) {
      absl::optional<XStatVisitor> element_id =
          produce_event->GetEventVisitor().GetStat(StatType::kElementId);
      if (!element_id.has_value()) continue;
      for (EventNode* produce_iterator : produce_event->GetChildren()) {
        if (IsIteratorEventType(produce_iterator->GetEventVisitor().Type())) {
          absl::optional<XStatVisitor> iterator_id =
              produce_iterator->GetEventVisitor().GetStat(StatType::kParentId);
          if (!iterator_id.has_value()) break;
          produce_iterator_map[{iterator_id->IntValue(),
                                element_id->IntValue()}]
              .push_back(produce_iterator);
          ++num_producers;
          break;
        }
      }
    }
  }
  VLOG(1) << num_producers << " producer iterators found.";
  uint64 num_matched = 0;
  for (HostEventType event_type :
       {HostEventType::kPrefetchConsume,
        HostEventType::kParallelInterleaveConsume,
        HostEventType::kParallelMapConsume, HostEventType::kMapAndBatchConsume,
        HostEventType::kParseExampleConsume}) {
    auto consume_event_list = gtl::FindOrNull(event_node_map_, event_type);
    if (!consume_event_list) continue;
    VLOG(1) << consume_event_list->size() << " "
            << GetHostEventTypeStr(event_type) << " events found.";
    for (auto& consume_event : *consume_event_list) {
      absl::optional<XStatVisitor> element_id =
          consume_event->GetEventVisitor().GetStat(StatType::kElementId);
      if (!element_id.has_value()) continue;
      if (consume_event->GetParents().empty()) continue;
      // consume_event is nested by consumer_iterator and does not have other
      // parents.
      EventNode* consume_iterator = consume_event->GetParents().at(0);
      if (!consume_iterator ||
          !IsIteratorEventType(consume_iterator->GetEventVisitor().Type())) {
        continue;
      }
      absl::optional<XStatVisitor> iterator_id =
          consume_iterator->GetEventVisitor().GetStat(StatType::kStepId);
      if (!iterator_id.has_value()) continue;
      if (auto produce_iterators = gtl::FindOrNull(
              produce_iterator_map, std::make_pair(iterator_id->IntValue(),
                                                   element_id->IntValue()))) {
        for (EventNode* produce_iterator : *produce_iterators) {
          consume_iterator->AddChild(produce_iterator);
          ++num_matched;
        }
      }
    }
  }
  VLOG(1) << num_matched << " consumer iterators matched.";
}

void EventForest::GroupEvents(
    const std::vector<int64>& user_defined_root_event_types) {
  ProcessTensorFlowLoop();
  ProcessWorker();
  ProcessUserDefinedRootEvents(user_defined_root_event_types);
  CreateEventGroups();
  MarkEagerlyExecutedGpuKernels();
  MarkEagerlyExecutedCpuTfOps();
  ProcessModelIds();
}

std::vector<InterThreadConnectInfo> CreateInterThreadConnectInfoList() {
  std::vector<InterThreadConnectInfo> connect_info_list = {
      {HostEventType::kExecutorStateProcess,
       HostEventType::kIteratorGetNextOp,
       {StatType::kStepId, StatType::kIterNum}},
      {HostEventType::kExecutorStateProcess,
       HostEventType::kIteratorGetNextAsOptionalOp,
       {StatType::kStepId, StatType::kIterNum}},
      {HostEventType::kKernelLaunch,
       HostEventType::kKernelExecute,
       {StatType::kCorrelationId}}};
  return connect_info_list;
}

void GroupTfEvents(XSpace* space, EventForest* event_forest) {
  if (CheckLoopOp(*space)) {
    // TODO(b/154510598): Support TF's loop ops.
    return;
  }
  std::vector<InterThreadConnectInfo> connect_info_list =
      CreateInterThreadConnectInfoList();
  event_forest->AddSpace(CreateTfXPlaneVisitor, space);
  event_forest->ConnectEvents(connect_info_list);
  event_forest->GroupEvents();
}

void GroupTfEvents(XSpace* space) {
  EventForest event_forest;
  GroupTfEvents(space, &event_forest);
}

}  // namespace profiler
}  // namespace tensorflow
