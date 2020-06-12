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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
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

struct ContextInfo {
  ContextInfo(int type, uint64 id) : type(type), id(id) {}
  int type;
  uint64 id;
};

// A wrapper for XEvent with parent and children pointers. Through these
// pointers, a tree of EventNode is formed.
class EventNode {
 public:
  // REQUIRED: all inputs should not be nullptr.
  EventNode(const XPlaneVisitor* plane, XLine* raw_line, XEvent* raw_event);

  EventNode(const EventNode& event_node);

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

  const XPlaneVisitor& GetPlaneVisitor() const { return *plane_; }

  const XEventVisitor& GetEventVisitor() const { return visitor_; }

  absl::optional<XStatVisitor> GetContextStat(int64 stat_type) const;

  void AddStepName(absl::string_view step_name);

  void SetIsEager(bool is_eager);

  // Returns true if this event is part of eagerly executed op.
  bool IsEager();

  bool IsNestedIn(EventNode* parent);

  // Returns the closest parent of the given event type.
  EventNode* FindParent(int64 event_type) const;

  absl::optional<ContextInfo> GetProducerContext() const {
    return producer_context_;
  }

  absl::optional<ContextInfo> GetConsumerContext() const {
    return consumer_context_;
  }

  void SetIsRoot(bool is_root) { is_root_ = is_root; }

  bool IsRoot() const { return is_root_; }

  bool IsAsync() const { return is_async_; }

  bool StartsBefore(const EventNode& other) const;

 private:
  const XPlaneVisitor* plane_;
  XEventVisitor visitor_;
  XLine* raw_line_;
  XEvent* raw_event_;
  EventNode* parent_ = nullptr;
  std::vector<EventNode*> children_;
  absl::optional<int64> group_id_;
  absl::optional<ContextInfo> producer_context_;
  absl::optional<ContextInfo> consumer_context_;
  bool is_root_ = false;
  bool is_async_ = false;
};

using EventNodeMap =
    absl::flat_hash_map<int64 /*event_type*/,
                        std::vector<std::unique_ptr<EventNode>>>;

using EventGroupNameMap = absl::flat_hash_map<int64 /*group_id*/, std::string>;

using EventList = std::vector<EventNode*>;

struct ContextGroup {
  EventNode* producer = nullptr;
  std::vector<EventNode*> consumers;
};

using ContextGroupMap = absl::flat_hash_map<
    int /*context_type*/,
    absl::flat_hash_map<uint64 /*context_id*/, ContextGroup>>;

// EventForest augments the input XSpace with the trace context. The trace
// context is created by stitching XEvents (1) using the nesting relationship
// within the same thread and (2) comparing the semantic arguments or using
// connect_info_list across threads. It also groups the events by the root
// events specified in root_event_types or marked by the semantic argument.
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
  void ConnectIntraThread(const XPlaneVisitor& visitor, XPlane* plane,
                          ContextGroupMap* context_groups);

  // Connects events across threads according to connect_info_list.
  void ConnectInterThread(
      const std::vector<InterThreadConnectInfo>& connect_info_list);

  void ProcessLegacyRootEvents(
      const std::vector<int64 /*EventType*/>& root_event_types);

  // Creates event groups and populates event_group_name_map_. If a TF loop is
  // used, each TF loop iteration becomes a root. Otherwise, top root events
  // (i.e., none of their ancestors is a root event) are used as roots. A new
  // group is created with all events reachable from a root.
  void CreateEventGroup();

  // Sets the is_eager stat to true for the eagerly executed GPU kernel events.
  void MarkEagerlyExecutedGpuKernels();

  // Sets the is_eager stat to true for the eagerly executed CPU TF op events.
  void MarkEagerlyExecutedCpuTfOps();

  // Processes the TF loops and registers the first TF executor event of each
  // iteraton to `tf_loop_root_events_`.
  void ProcessTensorFlowLoop();

  // Creates virtual events of HostEventType::kAsyncExecutorTraceContext. A
  // virtual event is created for every FunctionRun and the following eager ops
  // (e.g., for Keras callback).
  void CreateVirtualEventsForAsyncExecutor();

  EventNodeMap event_node_map_;
  std::vector<XPlaneVisitor> visitors_;
  EventGroupNameMap event_group_name_map_;
  EventList root_events_;
  EventList tf_loop_root_events_;
  int64 next_group_id_ = 0;
};

std::vector<InterThreadConnectInfo> CreateInterThreadConnectInfoList();

// Calls GroupEvents with connect_info_list and root_event_types specific to
// TensorFlow.
void GroupTfEvents(XSpace* space, EventGroupNameMap* event_group_name_map);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_GROUP_EVENTS_H_
