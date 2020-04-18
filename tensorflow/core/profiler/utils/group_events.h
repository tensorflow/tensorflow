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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_GROUP_EVENTS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_GROUP_EVENTS_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

// Information required to connect events across threads. The first two fields
// specify the event types of parent and child events. In addition to matching
// the event types, both events should have stats of the stat types specified
// in stat_types and their values should be the same.
struct InterThreadConnectInfo {
  int64 parent_event_type;
  int64 child_event_type;
  std::vector<int64> parent_stat_types;
  std::vector<int64> child_stat_types;
};

// A wrapper for XEvent with parent and children pointers. Through these
// pointers, a tree of EventNode is formed.
class EventNode {
 public:
  // REQUIRED: visitor and event should not be nullptr.
  explicit EventNode(const XPlaneVisitor* visitor, XEvent* event)
      : visitor_(visitor), event_(event) {
    DCHECK(visitor);
    DCHECK(event);
  }

  EventNode* GetParent() const { return parent_; }

  const std::vector<EventNode*>& GetChildren() const { return children_; }

  void AddChild(EventNode* child) {
    children_.push_back(child);
    child->parent_ = this;
  }

  absl::optional<int64> GetGroupId() const { return group_id_; }

  std::string GetGroupName() const;

  // Sets group_id for this node and its descendants.
  void PropagateGroupId(int64 group_id);

  const XPlaneVisitor& GetPlaneVisitor() const { return *visitor_; }

  const XEvent& GetEvent() const { return *event_; }

  const XStat* GetContextStat(int64 stat_type) const;

  void AddStepName(absl::string_view step_name);

  void SetIsEager(bool is_eager);

  // Returns true if this event is part of eagerly executed op.
  bool IsEager();

  bool IsNestedIn(EventNode* parent);

  // Returns the closest parent of the given event type.
  EventNode* FindParent(int64 event_type);

 private:
  const XPlaneVisitor* visitor_;
  XEvent* event_;
  EventNode* parent_ = nullptr;
  std::vector<EventNode*> children_;
  absl::optional<int64> group_id_;
};

using EventNodeMap =
    absl::flat_hash_map<int64 /*event_type*/,
                        std::vector<std::unique_ptr<EventNode>>>;

using VirtualEventContainer = std::vector<std::unique_ptr<XEvent>>;

using EventGroupNameMap = absl::flat_hash_map<int64 /*group_id*/, std::string>;

// Creates a forest of EventNode by stitching events in space using the nesting
// relationship within the same thread and connect_info_list across threads, and
// groups them by the root events.
class EventForest {
 public:
  EventForest(const std::vector<InterThreadConnectInfo>& connect_info_list,
              const std::vector<int64>& root_event_types,
              const std::function<XPlaneVisitor(const XPlane*)> visitor_factory,
              XSpace* space);

  const EventNodeMap& GetEventNodeMap() const { return event_node_map_; }

  const EventGroupNameMap& GetEventGroupNameMap() const {
    return event_group_name_map_;
  }

 private:
  // Creates an EventNode for each event in event_node_map and connect events
  // according to the nesting relationship within the thread.
  void ConnectIntraThread(const XPlaneVisitor& visitor, XPlane* plane);

  // Connects events across threads according to connect_info_list.
  void ConnectInterThread(
      const std::vector<InterThreadConnectInfo>& connect_info_list);

  // Creates event groups and populates event_group_name_map_. For each event of
  // each event type in root_event_types in order, if it was not grouped yet, a
  // new group is created with all the events reachable from the root event.
  void CreateEventGroup(
      const std::vector<int64 /*EventType*/>& root_event_types);

  // Sets the is_eager stat to true for the eagerly executed GPU kernel events.
  void MarkEagerlyExecutedGpuKernels();

  // Sets the is_eager stat to true for the eagerly executed CPU TF op events.
  void MarkEagerlyExecutedCpuTfOps();

  // Create virtual events of HostEventType::kHostTrainingLoopIteration and
  // event nodes for them. A virtual event is created for each iteration of the
  // host training loop and connected to the
  // HostEventType::kExecutorStateProcess event nodes of the iteration.
  void CreateVirtualEventsForHostTrainingLoop();

  // Create virutal events of HostEventType::kAsyncExecutorTraceContext and
  // event nodes for them. A virtual event is created for every FunctionRun and
  // the following eager ops (e.g., for Keras callback).
  void CreateVirtualEventsForAsyncExecutor();

  EventNodeMap event_node_map_;
  std::vector<XPlaneVisitor> visitors_;
  VirtualEventContainer virtual_event_container_;
  EventGroupNameMap event_group_name_map_;
};

std::vector<InterThreadConnectInfo> CreateInterThreadConnectInfoList();

// Calls GroupEvents with connect_info_list and root_event_types specific to
// TensorFlow.
void GroupTfEvents(XSpace* space, EventGroupNameMap* event_group_name_map);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_GROUP_EVENTS_H_
