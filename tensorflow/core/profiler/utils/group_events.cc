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
#include "absl/container/flat_hash_map.h"
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
}

// Returns event type if it is a KernelLaunch or KernelExecute event.
absl::optional<int64_t> GetKernelEventType(bool is_host_plane,
                                           const EventNode& event) {
  if (event.GetEventVisitor().GetStat(StatType::kCorrelationId).has_value()) {
    return is_host_plane ? HostEventType::kKernelLaunch
                         : HostEventType::kKernelExecute;
  }
  return absl::nullopt;
}

int64_t GetEventType(bool is_host_plane, const EventNode& event) {
  if (absl::optional<int64_t> event_type = event.GetEventVisitor().Type()) {
    return *event_type;
  } else if (absl::optional<int64_t> kernel_event_type =
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

bool HasFunctionRun(EventNode* event_node) {
  for (EventNode* child : event_node->GetChildren()) {
    if (child->GetEventVisitor().Type() == HostEventType::kFunctionRun) {
      return true;
    }
  }
  return false;
}

bool IsImplicitRootEvent(const XEventVisitor& event) {
  static const auto* const kImplicitRootEvents =
      new absl::flat_hash_set<int64_t>{
          HostEventType::kFunctionRun, HostEventType::kSessionRun,
          HostEventType::kRunGraph, HostEventType::kExecutorStateProcess};
  return event.Type().has_value() &&
         kImplicitRootEvents->contains(*event.Type());
}

void ProcessRootEvent(int64_t group_id, EventNode* root_event,
                      GroupMetadataMap* group_metadata_map) {
  root_event->PropagateGroupId(group_id, group_metadata_map);
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

struct ContextTypeAndId {
  int type;
  uint64 id;
};

absl::optional<ContextTypeAndId> GetLegacyProducerContext(
    const XEventVisitor& event) {
  absl::optional<ContextTypeAndId> type_and_id;
  absl::optional<int64_t> event_type = event.Type();
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
      default:
        break;
    }
  }
  return type_and_id;
}

absl::optional<ContextTypeAndId> GetLegacyConsumerContext(
    const XEventVisitor& event) {
  absl::optional<ContextTypeAndId> type_and_id;
  absl::optional<int64_t> event_type = event.Type();
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
  static const auto* const kRootEvents = new absl::flat_hash_set<int64_t>{
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

// Returns true if it has JAX-related events.
bool HasJaxEvent(const EventNodeMap& event_node_map) {
  return event_node_map.contains(HostEventType::kExecuteOnLocalDevices);
}

bool IsIteratorEventType(absl::optional<int64_t> event_type) {
  return event_type == HostEventType::kIterator ||
         event_type == HostEventType::kDeviceInputPipelineSecondIterator;
}

}  // namespace

// Returns true if TF's loop ops exist in the given XSpace's metadata.
bool CheckLoopOp(const XSpace& space) {
  for (const XPlane& plane : space.planes()) {
    for (const auto& event_metadata : plane.event_metadata()) {
      absl::optional<int64_t> event_type =
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
        root_level_ = stat.IntValue();
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
  root_level_ = root_level_ ? root_level_ : IsLegacyRootEvent(visitor_);

  if (producer_type.has_value() && producer_id.has_value()) {
    producer_context_ = {*producer_type, *producer_id};
  }
  if (consumer_type.has_value() && consumer_id.has_value()) {
    consumer_context_ = {*consumer_type, *consumer_id};
  }
}

absl::optional<XStatVisitor> EventNode::GetContextStat(
    int64_t stat_type) const {
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
  int64_t step_num = group_id_.value_or(0);
  if (absl::optional<XStatVisitor> stat = GetContextStat(StatType::kIterNum)) {
    step_num = stat->IntValue();
  } else if (absl::optional<XStatVisitor> stat =
                 GetContextStat(StatType::kStepNum)) {
    step_num = stat->IntValue();
  }
  absl::StrAppend(&name, step_num);
  return name;
}

XStat* EventNode::FindOrAddStatByType(int64_t stat_type) {
  const XStatMetadata* stat_metadata = plane_->GetStatMetadataByType(stat_type);
  DCHECK(stat_metadata != nullptr);
  return FindOrAddMutableStat(*stat_metadata, raw_event_);
}

void EventNode::SetGroupId(int64_t group_id) {
  group_id_ = group_id;
  FindOrAddStatByType(StatType::kGroupId)->set_int64_value(group_id);
}

void EventNode::PropagateGroupId(int64_t group_id,
                                 GroupMetadataMap* group_metadata_map) {
  std::queue<EventNode*> nodes;
  absl::flat_hash_set<EventNode*> seen = {this};
  nodes.push(this);
  while (!nodes.empty()) {
    EventNode* node = nodes.front();
    nodes.pop();
    absl::optional<int64_t> node_group_id = node->GetGroupId();
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

void EventNode::SetIsEager(bool is_eager) {
  FindOrAddStatByType(StatType::kIsEager)->set_int64_value(is_eager ? 1 : 0);
}

bool EventNode::IsCompiledFunc() const {
  auto is_func = visitor_.GetStat(StatType::kIsFunc);
  return !is_func || is_func->IntValue();
}

bool EventNode::IsEager() {
  /* Both eager mode (op-by-op) and non-eager mode (eager functions) of eager
   * executions are unified and forward to TF1 executor now. Therefore we will
   * check following conditions:
   */
  auto* node = FindParent(HostEventType::kEagerKernelExecute);
  if (node == nullptr) {
    // if current op is NOT scheduled under "EagerExecute", likely this is
    // from TF1, therefore not eager.
    return false;
  }

  // Otherwise, it is eager mode execution of an operation if and only if it is
  // not a eager mode execution of a compiled function.
  return !node->IsCompiledFunc();
}

const EventNode* EventNode::FindParent(int64_t event_type) const {
  return FindParentWithComparator(
      [event_type](const EventNode* node) {
        return node->GetEventVisitor().Type() == event_type;
      },
      this, /*include_self=*/true);
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
    const std::vector<int64_t>& parent_stat_types =
        connect_info.parent_stat_types;
    const std::vector<int64_t>* child_stat_types =
        &connect_info.child_stat_types;
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

// Returns whether a root event needs grouping.
bool RootNeedsGrouping(const EventNode* root) {
  // No grouping is needed if it is already grouped.
  if (root->GetGroupId().has_value()) return false;
  // If there is a parent node with the same root level, skip grouping at <root>
  // and later apply grouping at the parent node.
  // If there is a parent node with a different root level, apply grouping at
  // <root>, and later apply grouping at the parent node. Root events with
  // different levels are grouped separately.
  const EventNode* root_parent = FindParentWithComparator(
      [root](const EventNode* parent) {
        return parent->RootLevel() == root->RootLevel();
      },
      root,
      /*include_self=*/false);
  return root_parent == nullptr;
}

// Sorts root events based on root level and timestamp.
void SortRootEventList(EventList* event_list) {
  absl::c_sort(*event_list, [](const EventNode* e1, const EventNode* e2) {
    // If two root events have the same root level, the root event with an
    // earlier timestamp will be processed first. Otherwise, the event with a
    // larger root level will be processed first.
    return e1->RootLevel() == e2->RootLevel()
               ? *e1 < *e2
               : e1->RootLevel() > e2->RootLevel();
  });
}

void EventForest::CreateEventGroups() {
  // Create a group for each TF loop iteration in non-JAX profiles.
  int64_t group_id = 0;
  if (!HasJaxEvent(event_node_map_) && !tf_loop_root_events_.empty()) {
    for (EventNode* root_event : tf_loop_root_events_) {
      ProcessRootEvent(group_id++, root_event, &group_metadata_map_);
    }
    return;
  }

  // Iterate over all events and collect all root events.
  EventList root_events;
  for (const auto& typed_events : event_node_map_) {
    for (const auto& event : typed_events.second) {
      if (!event->RootLevel()) continue;
      absl::optional<XStatVisitor> step_id_stat =
          event->GetEventVisitor().GetStat(StatType::kStepId);
      // If this is a root event that associated with tf.data, skip.
      if (step_id_stat && tf_data_step_ids_.contains(step_id_stat->IntValue()))
        continue;
      root_events.push_back(event.get());
    }
  }

  SortRootEventList(&root_events);

  for (EventNode* root_event : root_events) {
    if (RootNeedsGrouping(root_event) &&
        // Ignores legacy TF root events for JAX profiles.
        (!HasJaxEvent(event_node_map_) ||
         !IsLegacyRootEvent(root_event->GetEventVisitor()))) {
      ProcessRootEvent(group_id++, root_event, &group_metadata_map_);
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

void EventForest::ProcessTfDataSteps() {
  const int64_t tf_data_event_types[] = {
      HostEventType::kTfDataCapturedFunctionRun,
      HostEventType::kTfDataCapturedFunctionRunAsync,
      HostEventType::kTfDataCapturedFunctionRunInstantiated,
      HostEventType::kTfDataCapturedFunctionRunWithBorrowedArgs};
  for (const int64_t tf_data_event_type : tf_data_event_types) {
    auto tf_data_events = gtl::FindOrNull(event_node_map_, tf_data_event_type);
    if (!tf_data_events) continue;
    for (const auto& tf_data_event : *tf_data_events) {
      absl::optional<XStatVisitor> step_id_stat =
          tf_data_event->GetEventVisitor().GetStat(StatType::kStepId);
      if (!step_id_stat) continue;
      tf_data_step_ids_.insert(step_id_stat->IntValue());
    }
  }
}

void EventForest::ProcessTensorFlowLoop() {
  struct TensorFlowLoopIteration {
    EventNode* first_event = nullptr;
    std::vector<EventNode*> events;
  };
  using TensorFlowLoop =
      absl::flat_hash_map<int64_t /*iter_num*/, TensorFlowLoopIteration>;
  absl::flat_hash_map<int64_t /*step_id*/, TensorFlowLoop> tf_loops;

  // Sort the TF executor events by TF function/session (step_id) and iter_num.
  auto executor_event_list =
      gtl::FindOrNull(event_node_map_, HostEventType::kExecutorStateProcess);
  if (!executor_event_list) return;
  for (auto& executor_event : *executor_event_list) {
    absl::optional<XStatVisitor> step_id_stat =
        executor_event->GetEventVisitor().GetStat(StatType::kStepId);
    absl::optional<XStatVisitor> iter_num_stat =
        executor_event->GetEventVisitor().GetStat(StatType::kIterNum);
    if (!step_id_stat || !iter_num_stat) continue;
    int64_t step_id = step_id_stat->IntValue();
    // Skip tf.data events.
    if (tf_data_step_ids_.contains(step_id)) continue;
    TensorFlowLoop& tf_loop = tf_loops[step_id];
    TensorFlowLoopIteration& iteration = tf_loop[iter_num_stat->IntValue()];
    if (!iteration.first_event || *executor_event < *iteration.first_event) {
      iteration.first_event = executor_event.get();
    }
    iteration.events.push_back(executor_event.get());
  }

  std::vector<const TensorFlowLoopIteration*> iters;
  for (const auto& step_id_and_tf_loop : tf_loops) {
    const TensorFlowLoop& tf_loop = step_id_and_tf_loop.second;
    // Filter out TF function/session without loops.
    if (tf_loop.size() == 1 && tf_loop.contains(0)) continue;
    for (const auto& iter_num_and_iter : tf_loop) {
      iters.push_back(&iter_num_and_iter.second);
    }
  }

  // Sort iterations based on timestamp of the first event in the iteration.
  absl::c_sort(iters, [](const auto& iter1, const auto& iter2) {
    return *iter1->first_event < *iter2->first_event;
  });

  // Register the first event of each iteration as a root event. Also, add the
  // other events of the iteration as child to the root event.
  for (const TensorFlowLoopIteration* iter : iters) {
    EventNode* root_event = iter->first_event;
    tf_loop_root_events_.push_back(root_event);
    for (EventNode* event : iter->events) {
      if (event == root_event) continue;
      root_event->AddChild(event);
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
      root_event->SetRootLevel(1);
    } else if (root_event) {
      // Add non-function eager ops as child.
      root_event->AddChild(eager_kernel_execute_event.get());
    }
  }
}

void EventForest::ProcessModelIds() {
  const int64_t model_id_event_type_list[] = {HostEventType::kSessionRun,
                                              HostEventType::kTfrtModelRun};
  for (const int64_t event_type : model_id_event_type_list) {
    auto event_list = gtl::FindOrNull(event_node_map_, event_type);
    if (!event_list) continue;
    for (const auto& event : *event_list) {
      auto group_id = event->GetGroupId();
      if (!group_id.has_value()) continue;
      absl::optional<XStatVisitor> model_id =
          event->GetEventVisitor().GetStat(StatType::kModelId);
      if (!model_id.has_value()) continue;
      group_metadata_map_[*group_id].model_id = model_id->ToString();
    }
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
  absl::flat_hash_map<
      std::pair<int64_t /*iterator_id*/, int64_t /*element_id*/>,
      std::vector<EventNode*>>
      produce_iterator_map;
  uint64 num_producers = 0;
  for (HostEventType event_type :
       {HostEventType::kPrefetchProduce,
        HostEventType::kParallelInterleaveProduce,
        HostEventType::kParallelMapProduce, HostEventType::kMapAndBatchProduce,
        HostEventType::kParseExampleProduce,
        HostEventType::kParallelBatchProduce}) {
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
        HostEventType::kParseExampleConsume,
        HostEventType::kParallelBatchConsume}) {
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

void EventForest::GroupEvents() {
  ProcessTfDataSteps();
  ProcessTensorFlowLoop();
  ProcessWorker();
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
