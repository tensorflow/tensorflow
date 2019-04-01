/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_COLOCATION_GRAPH_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_COLOCATION_GRAPH_H_

#include <unordered_map>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/port.h"

namespace tensorflow {

// Represents a node in the disjoint node forest and the
// accumulated constraints on the device used by that node.
class Member {
 public:
  Member() = default;

  Status SetParentAndSupportedDevices(const Node& node,
                                      const std::vector<DeviceType>& types);

  const DeviceNameUtils::ParsedName& requested_device_name() const {
    return requested_device_name_;
  }

  Status SetAssignedDeviceName(const string& device_name);
  Status SetResourceDeviceName(const Node& node);
  Status SetRequestedDeviceName(const Node& node);

  Status EnsureCompatibilityAcrossResourceEdge(
      const Node& src, const Member& src_root,
      const Node& dst, /*dst_root is this*/
      bool log_device_placement);

  const PrioritizedDeviceTypeVector& supported_device_types() const {
    return supported_device_types_;
  }

  // If `dry_run` is true, just sets `new_root` and `old_root` and does not
  // actually modify anything in the `tree`.
  static void Merge(std::vector<Member>* tree, int x_root, int y_root,
                    Member** new_root, Member** old_root, bool dry_run);

  // tree is non-const because we can change some `parent` pointers in some
  // members for more efficient future lookups. The vector itself is not
  // changed.
  static int FindRoot(std::vector<Member>* tree, int node_id);

  Status MergeDeviceNames(const Member& other, bool allow_soft_placement);

  // Updates this to contain the intersection of the device types in
  // this and "other". If the intersection is empty, returns false and does
  // not update this. Else returns true and updates this.
  bool MergeSupportedDevices(const Member& other);

  Status AssignDevice(const Node& node, bool allow_soft_placement);

  void set_possible_devices(std::vector<Device*>&& devices) {
    possible_devices_ = devices;
  }
  const std::vector<Device*>& possible_devices() { return possible_devices_; }

  // Returns a (parsed) device name that is based on requested_device_name()
  // but with potentially cleared device type and ID fields. A field is cleared
  // if the assigned_device_name does not specify it. If it does, the field
  // is not cleared because soft placement cannot violate assigned device names.
  DeviceNameUtils::ParsedName GetSoftDeviceName() const;

  // Same as GetSoftDeviceName but device type and device ID fields are not
  // cleared if resource device has them set.
  DeviceNameUtils::ParsedName GetPreferredSoftDeviceName() const;

  string DebugString() const;

 private:
  // The id of the node that is the parent of this one, or its own
  // id if it is a root. parent <= 0 indicates that this member is invalid.
  int parent_ = -1;

  // A proxy for the depth of the tree that is used to prefer
  // connecting smaller trees to larger trees when merging disjoint
  // sets.
  int rank_ = 0;

  // Once colocation groups have been formed, the Placer starts actually
  // choosing devices. All nodes in a group must be assigned to the same
  // device. Once we assigned the first device to some node in this group,
  // we set assigned_device_name_index to this device name's index in the
  // graph.
  // The `*_device_name_` fields will contain the parsed name of this device
  // and `possible_devices`, if computed, will contain just this device.
  // `assigned_device_name_index` is an optimization to avoid parsing and
  // comparing device names. The value of -1 signals that a single device
  // has not been chosen yet.
  int assigned_device_name_index_ = -1;

  // The merged form of the device requested for this node, with those of all of
  // its children. requested_device_name_ is always kept a specialization (i.e.
  // DeviceNameUtils::IsSpecialization) of assigned_device_name_. When no device
  // is requested, this field is set to assigned_device_name_.  As a
  // specialization of assigned_device_name_, requested_device_name_ represents
  // the most specific form of all assigned and requested devices of this node
  // and its children, if this node is a root. requested_device_name_ is used
  // to finally select devices for nodes.  We can override requested devices due
  // to resource colocation constraints but not assigned devices (unless soft
  // placement is on).
  // INVARIANT: requested_device_name_ is always kept a
  // DeviceNameUtils::IsSpecialization of assigned_device_name_ and
  // resource_device_name_. This makes requested_device_name_ the "accumulation
  // of all wishes" about the device.
  DeviceNameUtils::ParsedName requested_device_name_;

  // The merged form of the device assigned for this node, with
  // those of all of its children.
  // This field is used to raise errors due to unsatisfiable constraints.
  // Can be a partial specification.
  DeviceNameUtils::ParsedName assigned_device_name_;

  // The merged form of the requested resource device assigned for this node,
  // with those of all of its children.
  // This field is used to raise errors due to unsatisfiable constraints.
  // Can be a partial specification.
  // resource_device_name_ is initialized with user-requested device on nodes
  // producing resources, e.g. VarHandleOp.
  // For historical reasons, with soft placement enabled, Placer can "move"
  // resources (place resource producing ops on a device different from what
  // the user explicitly requested) when the colocation group of a resource
  // producing op contains ops that are not supported on the user-requested
  // resource device. A classic example of this is a sparse optimizer (only
  // supported on CPU) used on a GPU variable. In this case, the whole group
  // will be assigned to some device supported by all ops in the colocation
  // group. This is a surprising and unfortunate behavior because:
  //   1. Since soft_placement is on by default, users don't know that their
  //   variables are created on a different device than what they requested.
  //   Among other things, this can lead to surprising poor performance.
  //   2. Eager runtime cannot "move" resources. The same code can "work" when
  //   wrapped in tf.function but will fail when run eagerly.
  //   3. Extra complexity here to preserve these resource moving capabilities.
  DeviceNameUtils::ParsedName resource_device_name_;

  // The intersection of all device types supported by this node,
  // and those of all of its children, in priority order
  // of the preferred device.
  // It is possible that supported_device_types_ has an empty intersection with
  // requested/assigned/resource devices. We could have detected such cases
  // as soon as they happen and raise an error. Instead, for historical reasons,
  // we leave such error detection to the final device picking stage.
  PrioritizedDeviceTypeVector supported_device_types_;

  // If this node is a root, stores a list of Devices to which this node
  // and all of its children have been assigned, or nullptr if this
  // has not yet been computed.
  std::vector<Device*> possible_devices_;
};  // namespace

// This class maintains the connected components of a colocation
// constraint graph, and uses this information to assign a satisfying
// device placement to the nodes of the graph.
//
// The typical usage pattern is:
//
//   Graph graph = ...;
//   DeviceSet device_set = ...;
//   ColocationGraph colocation_graph(graph, device_set);
//
//   // Add all the nodes of the `graph` to the `colocation_graph`.
//   for (Node* node : graph.nodes()) {
//     TF_RETURN_IF_ERROR(colocation_graph.AddNode(*node));
//   }
//
//   // Add one or more colocation constraints.
//   Node node_1 = *graph.FindNodeId(...);
//   Node node_2 = *graph.FindNodeId(...);
//   TF_RETURN_IF_ERROR(colocation_graph.ColocateNodes(node_1, node_2));
//
//   // Assign devices based on the accumulated constraints.
//   for (Node* node : graph.nodes()) {
//     TF_RETURN_IF_ERROR(colocation_graph.AssignDevice(node));
//   }
//
// This implementation uses the Union-Find algorithm to efficiently maintain the
// connected components and incrementally adds edges via
// ColocationGraph::ColocateNodes() invocations.
//
// ColocationGraph does not assign any devices to graph nodes. The
// `log_device_placement` argument is used to log messages when requested
// device is ignored.
class ColocationGraph {
 public:
  ColocationGraph(const Graph* graph, const DeviceSet* device_set,
                  const Device* default_device, bool allow_soft_placement,
                  bool log_device_placement);

  // Adds each node of the Graph to this ColocationGraph as a singleton.
  //
  // NOTE: The implementation assumes that the ids of nodes passed to
  // this method are dense and zero-based; the memory used will be linear in
  // the largest node ID.
  // NOTE: If this method returns an error, *this is left in an undefined
  // state.
  Status ColocateAllNodes();

  Status ColocateResourceOrRefEdge(Node* src, Node* dst);

  Status ColocateResourceAndRefEdges();

  Status Initialize();

  Status ColocateNodeToGroup(
      std::unordered_map<StringPiece, const Node*, StringPieceHasher>*
          colocation_group_root,
      const Node* node, StringPiece colocation_group);

  // Merge the (possibly disjoint) sets containing nodes "x" and
  // "y". Returns OK if the all nodes in the union of these sets can
  // be placed on the same device type.
  //
  // If this method returns an error, *this is unchanged.
  Status ColocateNodes(const Node& x, const Node& y);

  // This overload of ColocateNodes() allows a caller to provide the root node
  // ids for the two nodes. For large graphs, this noticeably reduces the
  // graph load time.
  // If this method returns an error, *this is unchanged.
  Status ColocateNodes(const Node& x, int x_root, const Node& y, int y_root);

  // Limits the possible devices of `node`'s colocation group to the device
  // to which `node` is assigned. This makes sure that all nodes in this
  // colocation group will be assigned to the same device. Without this
  // explicit restriction, heuristics can choose a different possible device
  // for other nodes in the group.
  Status LimitToAssignedDevice(const Node& node);

  // For the given node, subject to the constraints previously given
  // to this ColocationGraph, set its assigned_device_name. Returns OK
  // if a satisfying device can be found, otherwise an error.
  //
  // Note: This method returns a pointer to a field within members_.
  // The caller must not use the returned pointer after there is any possibility
  // that the members_[i].possible_devices field has been modified.
  Status GetDevicesForNode(Node* node,
                           const std::vector<Device*>** possible_devices);

  void GetSoftDeviceCandidates(const Node& node, const Member& root_member,
                               int root_id,
                               std::vector<Device*>* possible_devices);

  Status InitializeMembers();

  string DebugString();

  // Returns debugging info for the node referred to by 'node_root'.
  string DebugInfo(const int node_root);

  Status InitializeMemberWithAssignedDevice(const string& assigned_device_name,
                                            const string& node_type,
                                            Member* member);

  Status InitializeMember(const Node& node, Member* member);

  // Returns the root node of the disjoint tree to which the node with the
  // given id is connected.
  int FindRoot(int node_id) { return Member::FindRoot(&members_, node_id); }

  const Graph* const graph_;  // Not owned.
  std::vector<Member> members_;
  const DeviceSet* device_set_;  // Not owned.
  const std::vector<DeviceType> device_types_;
  const Device* default_device_;
  const bool allow_soft_placement_;
  const bool log_device_placement_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_COLOCATION_GRAPH_H_
