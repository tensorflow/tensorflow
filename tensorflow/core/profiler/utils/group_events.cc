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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
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
  AddOrUpdateIntStat(*visitor.GetStatMetadataId(StatType::kGroupId), group_id,
                     event);
}

using VirtualEventNodeMap =
    absl::flat_hash_map<int64 /*step_id*/,
                        absl::flat_hash_map<int64 /*iter_num*/, EventNode*>>;

std::unique_ptr<XEvent> CreateVirtualEvent(const XStat& step_id_stat,
                                           const XStat& iter_num_stat) {
  auto virtual_event = absl::make_unique<XEvent>();
  *virtual_event->add_stats() = step_id_stat;
  *virtual_event->add_stats() = iter_num_stat;
  return virtual_event;
}

bool NeedsVirtualEventsForHostTrainingLoop(
    const std::vector<int64 /*EventType*/>& root_event_types) {
  return std::find(root_event_types.begin(), root_event_types.end(),
                   HostEventType::kHostTrainingLoopIteration) !=
         root_event_types.end();
}

bool NeedsVirtualEventsForAsyncExecutor(
    const std::vector<int64 /*EventType*/>& root_event_types) {
  return std::find(root_event_types.begin(), root_event_types.end(),
                   HostEventType::kAsyncExecutorTraceContext) !=
         root_event_types.end();
}

bool HasFunctionRun(EventNode* event_node) {
  for (EventNode* child : event_node->GetChildren()) {
    if (child->GetPlaneVisitor().GetEventType(child->GetEvent()) ==
        HostEventType::kFunctionRun) {
      return true;
    }
  }
  return false;
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
    XStatVisitor stat(visitor_, graph_type_stat);
    name_parts.push_back(stat.ToString());
  }
  int64 step_num = group_id_.value_or(0);
  if (const XStat* step_num_stat = GetContextStat(StatType::kStepNum)) {
    step_num = step_num_stat->int64_value();
  }
  if (const XStat* iter_num_stat = GetContextStat(StatType::kIterNum)) {
    step_num = iter_num_stat->int64_value();
  }
  name_parts.push_back(absl::StrCat(step_num));
  return absl::StrJoin(name_parts, " ");
}

void EventNode::PropagateGroupId(int64 group_id) {
  group_id_ = group_id;
  SetGroupId(*visitor_, group_id, event_);
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
  AddOrUpdateStrStat(*visitor_->GetStatMetadataId(StatType::kStepName),
                     step_name, event_);
}

void EventNode::SetIsEager(bool is_eager) {
  AddOrUpdateIntStat(*visitor_->GetStatMetadataId(StatType::kIsEager),
                     is_eager ? 1 : 0, event_);
}

bool EventNode::IsEager() {
  // It is eagerly executed if its trace context includes the EagerKernelExecute
  // event (which may execute an op eagerly or through the TF executor) but not
  // the TF executor event.
  return FindParent(HostEventType::kExecutorStateProcess) == nullptr &&
         FindParent(HostEventType::kEagerKernelExecute) != nullptr;
}

bool EventNode::IsNestedIn(EventNode* parent) {
  return parent && IsNested(GetEvent(), parent->GetEvent());
}

EventNode* EventNode::FindParent(int64 event_type) {
  if (parent_) {
    if (GetEventType(parent_->GetPlaneVisitor(), parent_->GetEvent()) ==
        event_type) {
      return parent_;
    }
    return parent_->FindParent(event_type);
  }
  return nullptr;
}

void EventForest::ConnectIntraThread(const XPlaneVisitor& visitor,
                                     XPlane* plane) {
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
          const XStat* stat = parent_event_node->GetContextStat(stat_type);
          if (!stat) break;
          stats.push_back(stat->value_case() == stat->kInt64Value
                              ? stat->int64_value()
                              : stat->uint64_value());
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
          const XStat* stat = child_event_node->GetContextStat(stat_type);
          if (!stat) break;
          stats.push_back(stat->value_case() == stat->kInt64Value
                              ? stat->int64_value()
                              : stat->uint64_value());
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
  int64 next_group_id = 0;
  for (int64 root_event_type : root_event_types) {
    if (auto root_event_node_list =
            gtl::FindOrNull(event_node_map_, root_event_type)) {
      for (const auto& root_event_node : *root_event_node_list) {
        // Skip if it already belongs to a group.
        if (root_event_node->GetGroupId()) continue;
        int64 group_id = next_group_id++;
        root_event_node->PropagateGroupId(group_id);
        std::string group_name = root_event_node->GetGroupName();
        // TODO(jihochoi): change event name instead.
        root_event_node->AddStepName(group_name);
        event_group_name_map_[group_id] = std::move(group_name);
      }
      // Only use the first root event type found.
      if (!root_event_node_list->empty()) break;
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

void EventForest::CreateVirtualEventsForHostTrainingLoop() {
  VirtualEventNodeMap virtual_event_node_map;
  auto executor_event_node_list =
      gtl::FindOrNull(event_node_map_, HostEventType::kExecutorStateProcess);
  if (!executor_event_node_list) return;
  for (auto& executor_event_node : *executor_event_node_list) {
    const XStat* step_id_stat =
        executor_event_node->GetContextStat(StatType::kStepId);
    const XStat* iter_num_stat =
        executor_event_node->GetContextStat(StatType::kIterNum);
    if (!step_id_stat || !iter_num_stat) continue;
    int64 step_id = step_id_stat->int64_value();
    int64 iter_num = iter_num_stat->int64_value();
    // Process the event with nonzero iter_num only to filter out the events
    // related to tf.data.
    // TODO(jihochoi): Filter out tf.data events more reliably.
    if (!iter_num) continue;
    EventNode*& virtual_event_node = virtual_event_node_map[step_id][iter_num];
    if (!virtual_event_node) {
      std::unique_ptr<XEvent> new_virtual_event =
          CreateVirtualEvent(*step_id_stat, *iter_num_stat);
      auto new_virtual_event_node = absl::make_unique<EventNode>(
          &executor_event_node->GetPlaneVisitor(), new_virtual_event.get());
      // virtual_event_container_ keeps new_virtual_event alive.
      virtual_event_container_.push_back(std::move(new_virtual_event));
      virtual_event_node = new_virtual_event_node.get();
      // event_node_map_ keeps new_virtual_event_node alive.
      event_node_map_[HostEventType::kHostTrainingLoopIteration].push_back(
          std::move(new_virtual_event_node));
    }
    virtual_event_node->AddChild(executor_event_node.get());
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
      auto new_virtual_event = absl::make_unique<XEvent>();
      auto new_virtual_event_node = absl::make_unique<EventNode>(
          &eager_kernel_execute_event_node->GetPlaneVisitor(),
          new_virtual_event.get());
      // virtual_event_container_ keeps new_virtual_event alive.
      virtual_event_container_.push_back(std::move(new_virtual_event));
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
  visitors_.reserve(space->planes_size());
  for (auto& plane : *space->mutable_planes()) {
    CreateStatMetadata(&plane);
    visitors_.push_back(visitor_factory(&plane));
    ConnectIntraThread(visitors_.back(), &plane);
  }
  ConnectInterThread(connect_info_list);
  if (NeedsVirtualEventsForHostTrainingLoop(root_event_types)) {
    CreateVirtualEventsForHostTrainingLoop();
  }
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
       HostEventType::kSessionRun, HostEventType::kRunGraph,
       HostEventType::kHostTrainingLoopIteration});
  EventForest event_forest(connect_info_list, root_event_types,
                           CreateTfXPlaneVisitor, space);
  if (event_group_name_map) {
    *event_group_name_map = event_forest.GetEventGroupNameMap();
  }
}

}  // namespace profiler
}  // namespace tensorflow
