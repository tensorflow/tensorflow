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

#ifndef XLA_TSL_PROFILER_UTILS_GROUP_EVENTS_H_
#define XLA_TSL_PROFILER_UTILS_GROUP_EVENTS_H_

#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

// Information required to connect events across threads. The first two fields
// specify the event types of parent and child events. In addition to matching
// the event types, both events should have stats of the stat types specified
// in stat_types and their values should be the same.
struct InterThreadConnectInfo {
  int64_t parent_event_type;
  int64_t child_event_type;
  std::vector<int64_t> parent_stat_types;
  std::vector<int64_t> child_stat_types;
};

struct GroupMetadata {
  std::string name;
  absl::flat_hash_set<int64_t> parents;
  absl::flat_hash_set<int64_t> children;
};

using GroupMetadataMap =
    absl::flat_hash_map<int64_t /*group_id*/, GroupMetadata>;

// A wrapper for XEvent with parent and children pointers. Through these
// pointers, a tree of EventNode is formed.
class EventNode {
 public:
  explicit EventNode(XEventVisitor visitor) : visitor_(std::move(visitor)) {}

  EventNode(const EventNode& event_node) = delete;
  EventNode& operator=(const EventNode&) = delete;

  const std::vector<EventNode*>& GetParents() const { return parents_; }

  const std::vector<EventNode*>& GetChildren() const { return children_; }

  void AddChild(EventNode* child) {
    children_.push_back(child);
    child->parents_.push_back(this);
  }

  std::optional<int64_t> GetGroupId() const { return group_id_; }

  std::string GetGroupName() const;

  void SetGroupId(int64_t group_id);

  // Sets group_id for this node and its descendants.
  void PropagateGroupId(int64_t group_id, GroupMetadataMap* group_metadata_map);

  const XEventVisitor& GetEventVisitor() const { return visitor_; }

  std::optional<XStatVisitor> GetContextStat(int64_t stat_type) const;

  void AddStepName(absl::string_view step_name);

  void SetIsEager(bool is_eager);

  // Returns true if this event is part of eagerly executed op.
  bool IsEager() const;

  bool IsNestedIn(EventNode* parent);

  // Returns the closest parent (including itself) of the given event type.
  const EventNode* FindParent(int64_t event_type) const;

  void SetRootLevel(int root_level) { root_level_ = root_level; }

  int RootLevel() const { return root_level_; }

  bool IsCompiledFunc() const;

  // Compare two EventNodes based on start timestamp.
  bool operator<(const EventNode& other) const {
    return GetEventVisitor().TimestampPs() <
           other.GetEventVisitor().TimestampPs();
  }

 private:
  XStat* FindOrAddStatByType(int64_t stat_type);

  XEventVisitor visitor_;
  std::vector<EventNode*> parents_;
  std::vector<EventNode*> children_;
  std::optional<int64_t> group_id_;
  // Root event level.
  // By default root_level_ is set to 0, which means it is not a root event.
  // Events with root_level_ greater than 0 are considered as root events.
  int root_level_ = 0;
};

using EventNodeMap =
    absl::flat_hash_map<int64_t /*event_type*/, std::deque<EventNode>>;

using EventList = std::vector<EventNode*>;

struct ContextGroup {
  std::vector<EventNode*> producers;
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
  void AddSpace(
      std::function<XPlaneVisitor(const tensorflow::profiler::XPlane*)>
          visitor_factory,
      tensorflow::profiler::XSpace* space);

  void AddPlanes(
      std::function<XPlaneVisitor(const tensorflow::profiler::XPlane*)>
          visitor_factory,
      const std::vector<tensorflow::profiler::XPlane*>& planes);

  void ConnectEvents(
      const std::vector<InterThreadConnectInfo>& connect_info_list = {});

  void ConnectTfDataEvents();

  void GroupEvents();

  const EventNodeMap& GetEventNodeMap() const { return event_node_map_; }

  const GroupMetadataMap& GetGroupMetadataMap() const {
    return group_metadata_map_;
  }

 private:
  void AddPlane(
      std::function<XPlaneVisitor(const tensorflow::profiler::XPlane*)>
          visitor_factory,
      tensorflow::profiler::XPlane* plane);

  // Creates an EventNode for each event in event_node_map and connect events
  // according to the nesting relationship within the thread.
  void ConnectIntraThread(tensorflow::profiler::XPlane* plane,
                          XPlaneVisitor* visitor,
                          ContextGroupMap* context_groups);

  // Connects events across threads according to connect_info_list.
  void ConnectInterThread(
      const std::vector<InterThreadConnectInfo>& connect_info_list);

  // Creates event groups and populates group_metadata_map. If a TF loop is
  // used, each TF loop iteration becomes a root. Otherwise, top root events
  // (i.e., none of their ancestors is a root event) are used as roots. A new
  // group is created with all events reachable from a root.
  void CreateEventGroups();

  // Sets the is_eager stat to true for the eagerly executed GPU kernel events.
  void MarkEagerlyExecutedGpuKernels();

  // Sets the is_eager stat to true for the eagerly executed CPU TF op events.
  void MarkEagerlyExecutedCpuTfOps();

  // Populate all the step ids that associated with tf.data pipeline.
  // Because FunctionRun is considered as root, but we want to exclude those
  // FunctionRuns from tf.data.
  void ProcessTfDataSteps();

  // Processes the TF loops and registers the first TF executor event of each
  // iteration to `tf_loop_root_events_`.
  void ProcessTensorFlowLoop();

  // Find the events of event_type which own ALL the given stat_types. If found,
  // apply the given function to the node. The query predicates are
  //     - The node's ContextStat contains stat_types(i.e. stat_types is a
  //     subset of ContextStat) *AND*
  //     - The node's event type in event_node_map_ is event_type.
  void FindEventNodeAndApply(
      int64_t event_type, const std::vector<int64_t>& stat_types,
      const std::function<void(EventNode&, const std::vector<uint64>&)>& cb);

  EventNodeMap event_node_map_;
  std::vector<XPlaneVisitor> visitors_;
  // std::deque for pointer stability.
  std::deque<std::pair<tensorflow::profiler::XPlane*, XPlaneVisitor>> planes_;
  // The "step" id (actually it is "function" id that are associated with
  // the tf.data pipeline.
  absl::flat_hash_set<int64_t> tf_data_step_ids_;
  EventList tf_loop_root_events_;
  GroupMetadataMap group_metadata_map_;
};

std::vector<InterThreadConnectInfo> CreateInterThreadConnectInfoList();

// Calls GroupEvents with connect_info_list and root_event_types specific to
// TensorFlow.
void GroupTfEvents(tensorflow::profiler::XSpace* space,
                   EventForest* event_forest);
void GroupTfEvents(tensorflow::profiler::XSpace* space);

// Returns true if the given space has TF's loop ops.
bool CheckLoopOp(const tensorflow::profiler::XSpace& space);

// Adds step names from GroupMetadataMap to "Steps" line in plane.
// The event name is updated when converted to trace events.
void AddGroupMetadataToStepEvents(const GroupMetadataMap& group_metadata_map,
                                  XLineBuilder& line);

void GroupHostAndPlanes(
    tensorflow::profiler::XSpace* space,
    const std::vector<tensorflow::profiler::XPlane*>& device_traces,
    EventForest* event_forest);

void GroupXplaneEvents(tensorflow::profiler::XPlane* plane,
                       const GroupMetadataMap& group_metadata_map);

void GroupTpuEventsOSS(
    tensorflow::profiler::XSpace* space,
    const std::vector<tensorflow::profiler::XPlane*>& device_traces,
    EventForest* event_forest);

}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_UTILS_GROUP_EVENTS_H_
