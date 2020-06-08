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
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

static const int64 kFunctionalOpEventTypes[] = {
    HostEventType::kCallOp,
    HostEventType::kNumericalGradientOpEvalRight,
    HostEventType::kNumericalGradientOpEvalLeft,
    HostEventType::kSymbolicGradientOp,
    HostEventType::kRemoteCallOp,
    HostEventType::kIfOp,
    HostEventType::kCaseOp,
    // TODO(b/154510598): Fix handling of the loop ops.
    // HostEventType::kWhileOpEvalCond,
    // HostEventType::kWhileOpStartBody,
    // HostEventType::kForOp,
    // HostEventType::kParallelForOp,
    // HostEventType::kForeverOp,
    HostEventType::kPartitionedCallOp,
};

// Creates stat metadata for the stats which may be added by grouping.
void CreateStatMetadata(XPlane* plane) {
  XPlaneBuilder builder(plane);
  builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId));
  builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kStepName));
  builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kIsEager));
}

// Returns event type if it is a KernelLaunch or KernelExecute event.
absl::optional<int64> GetKernelEventType(const XPlaneVisitor& visitor,
                                         const XEvent& event) {
  for (const auto& stat : event.stats()) {
    if (visitor.GetStatType(stat) == StatType::kCorrelationId) {
      // TODO(b/149095099): avoid string comparison.
      return visitor.Name() == kHostThreads ? HostEventType::kKernelLaunch
                                            : HostEventType::kKernelExecute;
    }
  }
  return absl::nullopt;
}

bool IsTfOpEvent(const XPlaneVisitor& visitor, const XEvent& event) {
  TfOp tf_op =
      ParseTfOpFullname(visitor.GetEventMetadata(event.metadata_id())->name());
  return tf_op.category == Category::kTensorFlow;
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
  } else if (IsTfOpEvent(visitor, event)) {
    return HostEventType::kTfOpRun;
  } else {
    return HostEventType::kUnknownHostEventType;
  }
}

void SetGroupId(const XPlaneVisitor& visitor, int64 group_id, XEvent* event) {
  AddOrUpdateIntStat(*visitor.GetStatMetadataId(StatType::kGroupId), group_id,
                     event);
}

void SetContextGroup(EventNode* event, ContextGroupMap* context_groups) {
  auto producer = event->GetProducerContext();
  if (producer.has_value()) {
    ((*context_groups)[producer->type][producer->id]).producer = event;
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
      EventNode* parent = group.producer;
      for (EventNode* child : group.consumers) {
        parent->AddChild(child);
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

bool NeedsVirtualEventsForAsyncExecutor(
    const std::vector<int64 /*EventType*/>& root_event_types) {
  return std::find(root_event_types.begin(), root_event_types.end(),
                   HostEventType::kAsyncExecutorTraceContext) !=
         root_event_types.end();
}

bool HasFunctionRun(EventNode* event_node) {
  for (EventNode* child : event_node->GetChildren()) {
    if (child->GetEventVisitor().Type() == HostEventType::kFunctionRun) {
      return true;
    }
  }
  return false;
}

void ProcessRootEvent(int64 group_id, EventNode* root_event,
                      EventGroupNameMap* event_group_name_map) {
  root_event->PropagateGroupId(group_id);
  std::string group_name = root_event->GetGroupName();
  // TODO(jihochoi): change event name instead.
  root_event->AddStepName(group_name);
  event_group_name_map->emplace(group_id, std::move(group_name));
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

}  // namespace

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
        producer_id = stat.IntValue();
        break;
      case StatType::kConsumerType:
        consumer_type = stat.IntValue();
        break;
      case StatType::kConsumerId:
        consumer_id = stat.IntValue();
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
  for (const EventNode* node = this; node != nullptr; node = node->parent_) {
    if (absl::optional<XStatVisitor> stat = node->visitor_.GetStat(stat_type)) {
      return stat;
    }
  }
  return absl::nullopt;
}

std::string EventNode::GetGroupName() const {
  std::string name;
  if (absl::optional<XStatVisitor> stat =
          GetContextStat(StatType::kGraphType)) {
    absl::StrAppend(&name, stat->StrOrRefValue(), " ");
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

void EventNode::PropagateGroupId(int64 group_id) {
  group_id_ = group_id;
  SetGroupId(*plane_, group_id, raw_event_);
  for (const auto& child : children_) {
    // Skip if it already belongs to a group. Some nodes may be added multiple
    // times as child (e.g., sometimes async ops are executed synchronously and
    // their nodes are added as child both in ConnectIntraThread and
    // ConnectInterThread).
    if (child->GetGroupId()) continue;
    child->PropagateGroupId(*group_id_);
  }
}

void EventNode::AddStepName(absl::string_view step_name) {
  AddOrUpdateStrStat(*plane_->GetStatMetadataId(StatType::kStepName), step_name,
                     raw_event_);
}

void EventNode::SetIsEager(bool is_eager) {
  AddOrUpdateIntStat(*plane_->GetStatMetadataId(StatType::kIsEager),
                     is_eager ? 1 : 0, raw_event_);
}

bool EventNode::IsEager() {
  // It is eagerly executed if its trace context includes the EagerKernelExecute
  // event (which may execute an op eagerly or through the TF executor) but not
  // the TF executor event.
  return FindParent(HostEventType::kExecutorStateProcess) == nullptr &&
         FindParent(HostEventType::kEagerKernelExecute) != nullptr;
}

EventNode* EventNode::FindParent(int64 event_type) const {
  if (parent_) {
    if (parent_->GetEventVisitor().Type() == event_type) {
      return parent_;
    }
    return parent_->FindParent(event_type);
  }
  return nullptr;
}

bool EventNode::StartsBefore(const EventNode& other) const {
  return GetEventVisitor().TimestampPs() <=
         other.GetEventVisitor().TimestampPs();
}

void EventForest::ConnectIntraThread(const XPlaneVisitor& visitor,
                                     XPlane* plane,
                                     ContextGroupMap* context_groups) {
  for (auto& line : *plane->mutable_lines()) {
    std::vector<EventNode*> parent_nodes;
    for (auto& event : *line.mutable_events()) {
      auto cur_node = absl::make_unique<EventNode>(&visitor, &line, &event);
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
      event_node_map_[GetEventType(visitor, event)].push_back(
          std::move(cur_node));
    }
  }
}

void EventForest::ConnectInterThread(
    const std::vector<InterThreadConnectInfo>& connect_info_list) {
  for (const auto& connect_info : connect_info_list) {
    absl::flat_hash_map<std::vector<int64>, EventNode*> connect_map;
    const std::vector<int64>& parent_stat_types =
        connect_info.parent_stat_types;
    const std::vector<int64>* child_stat_types = &connect_info.child_stat_types;
    if (child_stat_types->empty()) {
      child_stat_types = &parent_stat_types;
    }
    if (auto parent_event_node_list =
            gtl::FindOrNull(event_node_map_, connect_info.parent_event_type)) {
      for (const auto& parent_event_node : *parent_event_node_list) {
        std::vector<int64> stats;
        for (auto stat_type : parent_stat_types) {
          absl::optional<XStatVisitor> stat =
              parent_event_node->GetContextStat(stat_type);
          if (!stat) break;
          stats.push_back((stat->ValueCase() == XStat::kInt64Value)
                              ? stat->IntValue()
                              : stat->UintValue());
        }
        if (stats.size() == parent_stat_types.size()) {
          connect_map[stats] = parent_event_node.get();
        }
      }
    }
    if (auto child_event_node_list =
            gtl::FindOrNull(event_node_map_, connect_info.child_event_type)) {
      for (const auto& child_event_node : *child_event_node_list) {
        std::vector<int64> stats;
        for (auto stat_type : *child_stat_types) {
          absl::optional<XStatVisitor> stat =
              child_event_node->GetContextStat(stat_type);
          if (!stat) break;
          stats.push_back((stat->ValueCase() == XStat::kInt64Value)
                              ? stat->IntValue()
                              : stat->UintValue());
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

void EventForest::CreateEventGroup(
    const std::vector<int64 /*EventType*/>& root_event_types) {
  for (EventNode* root_event : tf_loop_root_events_) {
    ProcessRootEvent(next_group_id_++, root_event, &event_group_name_map_);
  }
  for (EventNode* root_event : root_events_) {
    ProcessRootEvent(next_group_id_++, root_event, &event_group_name_map_);
  }
  for (int64 root_event_type : root_event_types) {
    if (auto root_events = gtl::FindOrNull(event_node_map_, root_event_type)) {
      for (const auto& root_event : *root_events) {
        // Skip if it already belongs to a group.
        if (root_event->GetGroupId()) continue;
        ProcessRootEvent(next_group_id_++, root_event.get(),
                         &event_group_name_map_);
      }
      // Only use the first root event type found.
      if (!root_events->empty()) break;
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

void EventForest::CreateVirtualEventsForAsyncExecutor() {
  auto eager_kernel_execute_event_node_list =
      gtl::FindOrNull(event_node_map_, HostEventType::kEagerKernelExecute);
  if (!eager_kernel_execute_event_node_list) return;
  EventNode* virtual_event_node = nullptr;
  for (auto& eager_kernel_execute_event_node :
       *eager_kernel_execute_event_node_list) {
    if (HasFunctionRun(eager_kernel_execute_event_node.get())) {
      auto new_virtual_event_node =
          absl::make_unique<EventNode>(*eager_kernel_execute_event_node);
      virtual_event_node = new_virtual_event_node.get();
      // event_node_map_ keeps new_virtual_event_node alive.
      event_node_map_[HostEventType::kAsyncExecutorTraceContext].push_back(
          std::move(new_virtual_event_node));
    }
    if (virtual_event_node) {
      virtual_event_node->AddChild(eager_kernel_execute_event_node.get());
    }
  }
}

EventForest::EventForest(
    const std::vector<InterThreadConnectInfo>& connect_info_list,
    const std::vector<int64>& root_event_types,
    const std::function<XPlaneVisitor(const XPlane*)> visitor_factory,
    XSpace* space) {
  ContextGroupMap context_groups;
  visitors_.reserve(space->planes_size());
  for (auto& plane : *space->mutable_planes()) {
    CreateStatMetadata(&plane);
    visitors_.push_back(visitor_factory(&plane));
    ConnectIntraThread(visitors_.back(), &plane, &context_groups);
  }
  ConnectInterThread(connect_info_list);
  ConnectContextGroups(context_groups);
  ProcessTensorFlowLoop();
  if (NeedsVirtualEventsForAsyncExecutor(root_event_types)) {
    CreateVirtualEventsForAsyncExecutor();
  }
  CreateEventGroup(root_event_types);
  MarkEagerlyExecutedGpuKernels();
  MarkEagerlyExecutedCpuTfOps();
}

std::vector<InterThreadConnectInfo> CreateInterThreadConnectInfoList() {
  std::vector<InterThreadConnectInfo> connect_info_list = {
      {HostEventType::kFunctionRun,
       HostEventType::kExecutorStateProcess,
       {StatType::kStepId}},
      {HostEventType::kFunctionRun,
       HostEventType::kExecutorDoneCallback,
       {StatType::kStepId}},
      {HostEventType::kSessionRun,
       HostEventType::kExecutorStateProcess,
       {StatType::kStepId}},
      {HostEventType::kSessionRun,
       HostEventType::kExecutorDoneCallback,
       {StatType::kStepId}},
      {HostEventType::kRunGraph,
       HostEventType::kExecutorStateProcess,
       {StatType::kStepId}},
      {HostEventType::kRunGraph,
       HostEventType::kExecutorDoneCallback,
       {StatType::kStepId}},
      {HostEventType::kRunGraph,
       HostEventType::kRunGraphDone,
       {StatType::kStepId}},
      {HostEventType::kExecutorStateProcess,
       HostEventType::kIteratorGetNextOp,
       {StatType::kStepId, StatType::kIterNum}},
      {HostEventType::kExecutorStateProcess,
       HostEventType::kIteratorGetNextAsOptionalOp,
       {StatType::kStepId, StatType::kIterNum}},
      {HostEventType::kKernelLaunch,
       HostEventType::kKernelExecute,
       {StatType::kCorrelationId}},
      {HostEventType::kLocalExecutableExecuteOnLocalDevice,
       HostEventType::kLocalExecutableExecute,
       {StatType::kRunId}}};
  for (int64 event_type : kFunctionalOpEventTypes) {
    connect_info_list.push_back({event_type,
                                 HostEventType::kExecutorStateProcess,
                                 {StatType::kFunctionStepId},
                                 {StatType::kStepId}});
    connect_info_list.push_back({event_type,
                                 HostEventType::kExecutorDoneCallback,
                                 {StatType::kFunctionStepId},
                                 {StatType::kStepId}});
  }
  return connect_info_list;
}

void GroupTfEvents(XSpace* space, EventGroupNameMap* event_group_name_map) {
  if (!space) return;
  std::vector<InterThreadConnectInfo> connect_info_list =
      CreateInterThreadConnectInfoList();
  const std::vector<int64 /*EventType*/> root_event_types(
      {HostEventType::kTraceContext, HostEventType::kFunctionRun,
       HostEventType::kSessionRun, HostEventType::kRunGraph});
  EventForest event_forest(connect_info_list, root_event_types,
                           CreateTfXPlaneVisitor, space);
  if (event_group_name_map) {
    *event_group_name_map = event_forest.GetEventGroupNameMap();
  }
}

}  // namespace profiler
}  // namespace tensorflow
