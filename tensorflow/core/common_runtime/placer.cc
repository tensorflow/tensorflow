/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/placer.h"

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

namespace {

// We hoist the conversion from C-style string literal to StringPiece here,
// so that we can avoid the many repeated calls to strlen().
const StringPiece kColocationAttrNameStringPiece(kColocationAttrName);
const StringPiece kColocationGroupPrefixStringPiece(kColocationGroupPrefix);

// Returns a list of devices sorted by preferred type and then name
// from 'devices' whose type is in 'supported_device_types'.  This
// function searches the device types in 'supported_device_types' and
// returns the subset of devices that match.
std::vector<Device*> FilterSupportedDevices(
    const std::vector<Device*>& devices,
    const DeviceTypeVector& supported_device_types) {
  std::vector<Device*> filtered_devices;
  for (const DeviceType& d : supported_device_types) {
    for (Device* device : devices) {
      if (DeviceType(device->attributes().device_type()) == d) {
        filtered_devices.emplace_back(device);
      }
    }
  }

  auto device_sort = [](const Device* a, const Device* b) {
    auto a_priority = DeviceSet::DeviceTypeOrder(DeviceType(a->device_type()));
    auto b_priority = DeviceSet::DeviceTypeOrder(DeviceType(b->device_type()));
    // First sort by prioritized device type (higher is preferred) and
    // then by device name (lexicographically).
    if (a_priority != b_priority) {
      return a_priority > b_priority;
    }
    return StringPiece(a->name()) < StringPiece(b->name());
  };
  std::sort(filtered_devices.begin(), filtered_devices.end(), device_sort);
  return filtered_devices;
}

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
//   // Add all the nodes of graph to colocation_graph.
//   for (Node* node : graph.nodes()) {
//     TF_RETURN_IF_ERROR(colocation_graph.AddNode(*node));
//   }
//
//   // Add one or more colocation constraint.
//   Node node_1 = *graph.FindNodeId(...);
//   Node node_2 = *graph.FindNodeId(...);
//   TF_RETURN_IF_ERROR(colocation_graph.ColocateNodes(node_1, node_2));
//
//   // Assign devices based on the accumulated constraints.
//   for (Node* node : graph.nodes()) {
//     TF_RETURN_IF_ERROR(colocation_graph.AssignDevice(node));
//   }
//
// The implementation uses the union-find algorithm to maintain the
// connected components efficiently and incrementally as edges
// (implied by ColocationGraph::ColocateNodes() invocations) are added.
class ColocationGraph {
 public:
  ColocationGraph(Graph* graph, const DeviceSet* device_set,
                  bool allow_soft_placement)
      : graph_(graph),
        device_set_(device_set),
        device_types_(device_set->PrioritizedDeviceTypeList()),
        allow_soft_placement_(allow_soft_placement) {
    members_.resize(graph->num_node_ids());
  }

  // Adds each node of the Graph to this ColocationGraph as a singleton.
  //
  // NOTE: The implementation assumes that the ids of nodes passed to
  // this method are dense and zero-based; the memory used will be linear in
  // the largest node ID.
  // NOTE: If this method returns an error, *this is left in an undefined
  // state.
  Status ColocateAllNodes() {
    // This maps from a colocation group identifier to the 'root' of that
    // colocation group.  Note that the keys in this map are StringPiece; the
    // actual strings are stored under the NodeDef.  The lifetime of this map
    // is limited to this ColocateAllNodes() method, and no part of the
    // NodeDef trees are changed during the lifetime of this method, so using
    // StringPiece as a key is safe.
    //
    // Also, as a further optimization, we remove the "loc:@" prefix from
    // "class" attribute values, when they are used as keys in this table.
    // This allows us to use StringPiece values that refer to substrings of
    // 'string' values stored in NodeDef attribute lists, as well as StringPiece
    // values that refer to 'string' values from NodeDef::name(), without
    // performing any string allocations.
    std::unordered_map<StringPiece, const Node*, StringPieceHasher>
        colocation_group_root;

    for (Node* node : graph_->nodes()) {
      if (!node->IsOp()) {
        continue;
      }

      // When adding the node, identify whether it is part of a
      // colocation group.

      // This code is effectively the equivalent of GetNodeAttr() for a string
      // array, but it avoids all internal allocations (the allocation of the
      // backing store of the std::vector<string> as well as the copies of the
      // strings within it).  Instead, we combine the query of the colocation
      // attribute with the calls to ColocateNodeToGroup.
      bool found_spec = false;
      const AttrValue* attr_value =
          node->attrs().Find(kColocationAttrNameStringPiece);
      if (attr_value != nullptr && attr_value->has_list()) {
        for (const string& class_spec : attr_value->list().s()) {
          StringPiece spec(class_spec);
          if (str_util::ConsumePrefix(&spec,
                                      kColocationGroupPrefixStringPiece)) {
            found_spec = true;
            TF_RETURN_IF_ERROR(
                ColocateNodeToGroup(&colocation_group_root, node, spec));
          }
        }
      }

      if (!found_spec) {
        // If the node does not specify a colocation group, then use the
        // name of this node as the colocation group.
        TF_RETURN_IF_ERROR(
            ColocateNodeToGroup(&colocation_group_root, node, node->name()));
      }
    }

    return Status::OK();
  }

  Status ColocateNodeToGroup(
      std::unordered_map<StringPiece, const Node*, StringPieceHasher>*
          colocation_group_root,
      Node* node, StringPiece colocation_group) {
    const Node*& root_node = (*colocation_group_root)[colocation_group];
    if (root_node == nullptr) {
      // This is the first node of the colocation group, so
      // designate this node as the 'root' of that colocation group.
      root_node = node;
    } else {
      // Try to colocate the node with the root.  If there is an
      // error, return it.
      Status s = ColocateNodes(*node, *root_node);
      if (!s.ok()) {
        return AttachDef(s, *node);
      }
    }
    return Status::OK();
  }

  // Merge the (possibly disjoint) sets containing nodes "x" and
  // "y". Returns OK if the all nodes in the union of these sets can
  // be placed on the same device type.
  //
  // NOTE: If this method returns an error, *this is left in an undefined
  // state.
  Status ColocateNodes(const Node& x, const Node& y) {
    int x_root = FindRoot(x.id());
    int y_root = FindRoot(y.id());
    return ColocateNodes(x, x_root, y, y_root);
  }

  // This overload of ColocateNodes() allows a caller to provide the root node
  // ids for the two nodes. For large graphs, this noticeably reduces the
  // graph load time.
  Status ColocateNodes(const Node& x, int x_root, const Node& y, int y_root) {
    if (x_root == y_root) {
      return Status::OK();
    }

    DCHECK_EQ(x_root, FindRoot(x.id()));
    DCHECK_EQ(y_root, FindRoot(y.id()));

    Member& x_root_member = members_[x_root];
    Member& y_root_member = members_[y_root];

    // Merge the sets by swinging the parent pointer of the smaller
    // tree to point to the root of the larger tree. Together with
    // path compression in ColocationGraph::FindRoot, this ensures
    // that we do not experience pathological performance on graphs
    // such as chains.
    int new_root, old_root;
    if (x_root_member.rank < y_root_member.rank) {
      // The tree rooted at x_root is shallower, so connect it to
      // y_root. The rank of y_root is unchanged because its new
      // child has strictly less rank.
      x_root_member.parent = y_root;
      new_root = y_root;
      old_root = x_root;
    } else if (x_root_member.rank > y_root_member.rank) {
      // The tree rooted at y_root is shallower, so connect it to
      // x_root. The rank of x_root is unchanged because its new
      // child has strictly less rank.
      y_root_member.parent = x_root;
      new_root = x_root;
      old_root = y_root;
    } else {
      // Both trees have the same rank, so break the tie by choosing
      // x_root as the new root.
      y_root_member.parent = x_root;
      // Increment the rank of the tree rooted at x_root, because it
      // is now strictly deeper than before.
      ++x_root_member.rank;
      new_root = x_root;
      old_root = y_root;
    }

    Member& new_root_member = members_[new_root];
    Member& old_root_member = members_[old_root];

    // Merge the partial device specifications, and ensure that they are
    // compatible. NULL options_ is treated as allowing soft placement.
    // TODO(mrry): Consider enriching the error message by pointing
    // out which nodes have the explicit partial device
    // specifications that caused this conflict.
    Status s = DeviceNameUtils::MergeDevNames(&new_root_member.device_name,
                                              old_root_member.device_name,
                                              allow_soft_placement_);
    if (!s.ok()) {
      return errors::InvalidArgument("Cannot colocate nodes '", x.name(),
                                     "' and '", y.name(), ": ",
                                     s.error_message());
    }

    // Ensure that the common root has at least one supported device
    // type, by computing the intersection of
    // new_root_member.supported_device_types and
    // old_root_member.supported_device_types.
    MergeSupportedDevices(&new_root_member.supported_device_types,
                          old_root_member.supported_device_types);
    if (new_root_member.supported_device_types.empty()) {
      return errors::InvalidArgument(
          "Cannot colocate nodes '", x.name(), "' and '", y.name(),
          "' because no device type supports both of those nodes and the "
          "other nodes colocated with them.",
          DebugInfo(x_root), DebugInfo(y_root));
    }

    return Status::OK();
  }

  // For the given node, subject to the constraints previously given
  // to this ColocationGraph, set its assigned_device_name. Returns OK
  // if a satisfying device can be found, otherwise an error.
  //
  // Note: This method returns a pointer to a field within members_.
  // The caller must not use the returned pointer after there is any possibility
  // that the members_[i].possible_devices field has been modified.
  Status GetDevicesForNode(Node* node,
                           std::vector<Device*>** possible_devices) {
    *possible_devices = nullptr;
    const int node_root = FindRoot(node->id());
    if (!members_[node_root].possible_devices.empty()) {
      *possible_devices = &members_[node_root].possible_devices;
      return Status::OK();
    }

    // We have not yet computed the possible devices for the
    // colocated node set containing 'node', so we do so now using the
    // constraints on the root node.

    // "devices" will contain the set of feasible placements for the
    // colocated node set containing 'node'.
    std::vector<Device*> devices;
    if (DeviceNameUtils::HasSomeDetails(members_[node_root].device_name)) {
      // The root node has a (possibly partial) device
      // specification, so enumerate the physical devices that
      // conform to it.
      device_set_->FindMatchingDevices(members_[node_root].device_name,
                                       &devices);

      if (!devices.empty()) {
        // Filter devices into those that are compatible with the root
        // node (and its children).
        devices = FilterSupportedDevices(
            devices, members_[node_root].supported_device_types);
      }

      // Perform soft placement if allow_soft_placement_ is set.
      if (devices.empty() && allow_soft_placement_) {
        // The soft_device_name is the same as the node's device name
        // without specifying the device type or ID.
        DeviceNameUtils::ParsedName soft_device_name =
            members_[node_root].device_name;
        soft_device_name.type.clear();
        soft_device_name.has_type = false;
        soft_device_name.has_id = false;
        device_set_->FindMatchingDevices(soft_device_name, &devices);
        if (!devices.empty()) {
          devices = FilterSupportedDevices(
              devices, members_[node_root].supported_device_types);
        }
      }

      if (devices.empty()) {
        // Return an error when a physical device that matches an explicit
        // device specification is not found. This ensures that we don't
        // assign a node to GPU when the user wanted to force it on CPU.
        string debug_info = DebugInfo(node_root);

        DeviceNameUtils::ParsedName specified_device_name;
        if (DeviceNameUtils::ParseFullName(node->requested_device(),
                                           &specified_device_name) &&
            specified_device_name == members_[node_root].device_name) {
          // The specified device and merged set device match, and
          // will appear in the GraphDef (for debugging), so just
          // print the specified device.
          std::vector<Device*> devices_matching_nodedef;
          device_set_->FindMatchingDevices(specified_device_name,
                                           &devices_matching_nodedef);
          if (devices_matching_nodedef.empty()) {
            // Sometimes it is almost impossible to understand the problem
            // without a list of available devices.
            std::vector<string> device_names;
            for (const Device* device : device_set_->devices()) {
              device_names.push_back(device->name());
            }
            std::sort(device_names.begin(), device_names.end());

            return errors::InvalidArgument(
                "Operation was explicitly assigned to ",
                node->requested_device(), " but available devices are [ ",
                str_util::Join(device_names, ", "), " ]. Make sure ",
                "the device specification refers to a valid device.");
          } else if (specified_device_name.has_type) {
            return errors::InvalidArgument(
                "Could not satisfy explicit device specification '",
                node->requested_device(), "' because no supported kernel for ",
                specified_device_name.type, " devices is available.",
                debug_info, "\nRegistered kernels:\n",
                KernelsRegisteredForOp(node->type_string()));
          } else {
            return errors::InvalidArgument(
                "Could not satisfy explicit device specification '",
                node->requested_device(), debug_info);
          }
        } else {
          // The specified device may be a valid device but the
          // merged set device is different, so print both.
          return errors::InvalidArgument(
              "Could not satisfy explicit device specification '",
              node->requested_device(),
              "' because the node was colocated with a group of nodes that "
              "required incompatible device '",
              DeviceNameUtils::ParsedNameToString(
                  members_[node_root].device_name),
              "'", debug_info);
        }
      }
    } else {
      // The device is completely unspecified, so enumerate the devices that
      // support all of the nodes in the set.
      if (device_set_->devices().empty()) {
        return errors::Internal("No devices are registered");
      }
      devices = FilterSupportedDevices(
          device_set_->devices(), members_[node_root].supported_device_types);

      if (devices.empty()) {
        return errors::InvalidArgument(
            "Node had no OpKernel registered to support this operation: ",
            "Operation was ", node->type_string(), " and inputs were ",
            DataTypeVectorString(node->input_types()), DebugInfo(node_root));
      }
    }

    // Cache the result of the possible devices for this node group.
    members_[node_root].possible_devices = std::move(devices);
    *possible_devices = &members_[node_root].possible_devices;
    return Status::OK();
  }

  Status InitializeMembers() {
    for (Node* node : graph_->nodes()) {
      if (!node->IsOp()) {
        continue;
      }
      Status status = InitializeMember(*node, &members_[node->id()]);
      if (!status.ok()) {
        return AttachDef(status, *node);
      }
    }
    return Status::OK();
  }

  // Represents a node in the disjoint node set forest, and the
  // accumulated constraints on the device used by that node.
  struct Member {
    Member() = default;
    // The id of the node that is the parent of this one, or its own
    // id if it is a root. parent <= 0 indicates that this member is invalid.
    int parent = -1;

    // A proxy for the depth of the tree that is used to prefer
    // connecting smaller trees to larger trees when merging disjoint
    // sets.
    int rank = 0;

    // The intersection of all device types supported by this node,
    // and those of all of its children, in priority order
    // of the preferred device.
    DeviceTypeVector supported_device_types;

    // The merged form of the device requested for this node, with
    // those of all of its children.
    DeviceNameUtils::ParsedName device_name;

    // If this node is a root, stores a list of Devices to which this node
    // and all of its children have been assigned, or nullptr if this
    // has not yet been computed.
    std::vector<Device*> possible_devices;
  };

  // Returns debugging info for the node referred to by 'node_root'.
  string DebugInfo(const int node_root) {
    string text(
        "\nColocation Debug Info:\n"
        "Colocation group had the following types and devices: ");

    // If this node is part of a colocation group, then we want to
    // collect the mapping of ops to supported devices, so that
    // the user can see why an unsatisfiable placement occurred.

    std::unordered_map<string, string> type_to_devices;
    std::vector<const Node*> colocation_nodes;
    int num_nodes_found = 0;

    for (const Node* node : graph_->nodes()) {
      if (!node->IsOp()) {
        continue;
      }
      int id = node->id();
      if (FindRoot(id) != node_root) {
        continue;
      }
      ++num_nodes_found;
      colocation_nodes.push_back(node);
      const string& op_type = node->type_string();
      string devices_registered;
      for (const auto& device_type : members_[id].supported_device_types) {
        strings::StrAppend(&devices_registered, DeviceTypeString(device_type),
                           " ");
      }

      type_to_devices[op_type] = std::move(devices_registered);
    }

    for (const auto& td : type_to_devices) {
      strings::StrAppend(&text, "\n", td.first, ": ", td.second);
    }
    strings::StrAppend(&text,
                       "\n\nColocation members and user-requested devices:");
    for (const Node* node : colocation_nodes) {
      strings::StrAppend(&text, "\n  ", node->name(), " (", node->type_string(),
                         ") ", node->requested_device());
    }
    strings::StrAppend(&text, "\n");

    if (num_nodes_found <= 1) {
      text.clear();
    }
    return text;
  }

  Status InitializeMember(const Node& node, Member* member) {
    const int id = node.id();
    DCHECK_GE(id, 0);
    member->parent = id;
    TF_RETURN_IF_ERROR(SupportedDeviceTypesForNode(
        device_types_, node.def(), &member->supported_device_types));

    if (node.has_assigned_device_name()) {
      // This node has already been assigned to a device, so we
      // respect this placement, after sanity-checking it.  The
      // device_name and supported_device_types for this node reflect
      // the assigned device, so any nodes colocated with this node
      // will be assigned to the same device (assuming this is
      // possible).
      // NOTE: Since any assignment must have been performed by
      // the TensorFlow runtime, we consider errors in this branch to
      // be INTERNAL.
      const string& assigned_device_name = node.assigned_device_name();
      if (!DeviceNameUtils::ParseFullName(assigned_device_name,
                                          &member->device_name)) {
        return errors::Internal("Malformed assigned device '",
                                assigned_device_name, "'");
      }
      const Device* assigned_device =
          device_set_->FindDeviceByName(assigned_device_name);
      if (assigned_device == nullptr) {
        return errors::Internal("Assigned device '", assigned_device_name,
                                "' does not match any device");
      }

      for (const DeviceType& d : member->supported_device_types) {
        if (DeviceType(assigned_device->attributes().device_type()) == d) {
          return Status::OK();
        }
      }

      return errors::Internal("Assigned device '", assigned_device_name,
                              "' does not have registered OpKernel support "
                              "for ",
                              node.type_string());
    } else {
      // This node has not yet been assigned to a device, so we
      // calculate any constraints due to the set of registered
      // kernels and any (partial) user-provided device specification
      // in the NodeDef.

      // If no kernels are registered for this op type, fail with an error.
      if (member->supported_device_types.empty()) {
        std::set<string> registered_device_types;
        for (Device* d : device_set_->devices()) {
          registered_device_types.insert(d->device_type());
        }
        return errors::InvalidArgument(
            "No OpKernel was registered to support Op '", node.type_string(),
            "' with these attrs.  Registered devices: [",
            str_util::Join(registered_device_types, ","),
            "], Registered kernels:\n",
            KernelsRegisteredForOp(node.type_string()));
      }

      // If the NodeDef contains a device, then we interpret it as a
      // (partial) device specification.
      if (!node.requested_device().empty()) {
        // The user has specified a device in the NodeDef, try to find a
        // valid device matching their specification in the set of
        // devices.
        // NOTE: The full name may specify a device that is not in
        // n.supported_device_types(), but we check that in AssignDevice().
        if (!DeviceNameUtils::ParseFullName(node.requested_device(),
                                            &member->device_name)) {
          return errors::InvalidArgument("Malformed device specification '",
                                         node.requested_device(), "'");
        }
      }
    }
    return Status::OK();
  }

  // Updates target to contain the intersection of the device types in
  // "target" and "other".
  static void MergeSupportedDevices(DeviceTypeVector* target,
                                    const DeviceTypeVector& other) {
    DeviceTypeVector temp = *target;
    target->clear();

    // Iterate in priority order.
    for (const DeviceType& device_type : temp) {
      bool found = false;
      for (const DeviceType& other_device_type : other) {
        if (device_type == other_device_type) {
          found = true;
          break;
        }
      }
      if (found) {
        target->push_back(device_type);
      }
    }
  }

  // Returns the root node of the disjoint tree to which the node with the
  // given id is connected.
  int FindRoot(int node_id) {
    Member& member = members_[node_id];

    int parent = member.parent;
    DCHECK_GE(parent, 0);

    if (parent != node_id) {
      // NOTE: Compress paths from node_id to its root, so that future
      // calls to FindRoot and ColocateNodes are more efficient.
      int root = FindRoot(parent);
      if (parent != root) {
        parent = root;
        member.parent = root;
      }
    }

    DCHECK_GE(parent, 0);
    return parent;
  }

  // Ensures that the devices of 'dst's resource and reference match the device
  // specified for 'src', which is an input of 'dst' with a partially or fully
  // specified device.
  //
  // TODO(akshayka): Consider adding a way to whitelist ops that want to be
  // exempt from this check.
  Status VerifyResourceAndRefInputsCanBeColocated(
      const Node* dst, const Node* src,
      const DeviceNameUtils::ParsedName& src_parsed_name) {
    std::vector<const Edge*> edges;
    TF_RETURN_IF_ERROR(dst->input_edges(&edges));
    for (const Edge* edge : edges) {
      DataType input_type = dst->input_type(edge->dst_input());
      if (input_type == DT_RESOURCE || IsRefType(input_type)) {
        const Node* input_node = edge->src();
        if (input_node == src) {
          continue;
        }
        const auto& input_root = members_[FindRoot(input_node->id())];
        const auto& input_parsed_name = input_root.device_name;
        if (DeviceNameUtils::HasSomeDetails(input_parsed_name) &&
            !DeviceNameUtils::AreCompatibleDevNames(input_parsed_name,
                                                    src_parsed_name)) {
          return AttachDef(
              errors::InvalidArgument(
                  "Could not colocate node with its "
                  "resource and reference inputs; devices ",
                  DeviceNameUtils::ParsedNameToString(input_parsed_name),
                  " and ", DeviceNameUtils::ParsedNameToString(src_parsed_name),
                  " are not compatible."),
              *dst);
        }
      }
    }
    return Status::OK();
  }

  Graph* const graph_;  // Not owned.
  std::vector<Member> members_;
  const DeviceSet* device_set_;  // Not owned.
  const std::vector<DeviceType> device_types_;
  const bool allow_soft_placement_;
};

// Returns true if the node has no inputs and produces outputs
// that are consumed by a single node.
//
// TODO(vrv): Currently this handles only nodes with one output, but
// this could be extended to handle the case where a node has many
// outputs that are connected to nodes in the same colocation group.
bool IsGeneratorNode(const Node* node) {
  return node->num_inputs() == 0 && node->num_outputs() == 1 &&
         !IsRefType(node->output_type(0));
}

}  // namespace

Placer::Placer(Graph* graph, const DeviceSet* devices,
               const SessionOptions* options)
    : graph_(graph),
      devices_(devices),
      options_(options),
      log_device_placement_(options != nullptr &&
                            options->config.log_device_placement()) {}

Placer::Placer(Graph* graph, const DeviceSet* devices)
    : Placer(graph, devices, nullptr) {}

Placer::~Placer() {}

Status Placer::Run() {
  if (devices_->devices().empty()) {
    return errors::FailedPrecondition("No devices are registered");
  }

  ColocationGraph colocation_graph(
      graph_, devices_,
      options_ == nullptr || options_->config.allow_soft_placement());

  TF_RETURN_IF_ERROR(colocation_graph.InitializeMembers());

  // 1. First add all of the nodes. Note that steps (1) and (2)
  // requires two passes over the nodes because the graph (and hence
  // the constraints) may not be acyclic.
  TF_RETURN_IF_ERROR(colocation_graph.ColocateAllNodes());

  // 2. Enumerate the constraint edges, and use them to update the disjoint
  // node set.

  // If `node` has an input edge with reference type, add an
  // edge from the source of that edge to `node`.
  for (const Edge* edge : graph_->edges()) {
    if (edge->IsControlEdge()) {
      continue;
    }
    Node* src = edge->src();
    Node* dst = edge->dst();
    DataType input_type = dst->input_type(edge->dst_input());
    if (input_type == DT_RESOURCE || IsRefType(input_type)) {
      int src_root_id = colocation_graph.FindRoot(src->id());
      int dst_root_id = colocation_graph.FindRoot(dst->id());
      auto& src_root = colocation_graph.members_[src_root_id];
      auto& dst_root = colocation_graph.members_[dst_root_id];
      // If both the source node and this node have partially
      // specified a device, then 'node's device should be
      // cleared: the reference edge forces 'node' to be on the
      // same device as the source node.
      const auto& source_parsed_name = src_root.device_name;
      const auto& dest_parsed_name = dst_root.device_name;
      if (DeviceNameUtils::HasSomeDetails(source_parsed_name) &&
          DeviceNameUtils::HasSomeDetails(dest_parsed_name)) {
        // Ignore a specified device for 'dst' if the two names were
        // incompatible.
        if (!DeviceNameUtils::AreCompatibleDevNames(source_parsed_name,
                                                    dest_parsed_name)) {
          TF_RETURN_IF_ERROR(
              colocation_graph.VerifyResourceAndRefInputsCanBeColocated(
                  dst, src, source_parsed_name));
          if (log_device_placement_) {
            LOG(INFO) << "Ignoring device specification "
                      << DeviceNameUtils::ParsedNameToString(dest_parsed_name)
                      << " for node '" << dst->name()
                      << "' because the input edge from '" << src->name()
                      << "' is a reference connection and already has a device "
                         "field set to "
                      << DeviceNameUtils::ParsedNameToString(
                             source_parsed_name);
          }

          // Make 'dst' colocated with the source
          dst_root.device_name = source_parsed_name;
        } else {
          bool source_subset_of_dest = DeviceNameUtils::IsSpecification(
              source_parsed_name, dest_parsed_name);
          bool dest_subset_of_source = DeviceNameUtils::IsSpecification(
              dest_parsed_name, source_parsed_name);

          if (source_subset_of_dest && !dest_subset_of_source) {
            src_root.device_name = dest_parsed_name;
          } else {
            dst_root.device_name = source_parsed_name;
          }
        }
      }

      Status status =
          colocation_graph.ColocateNodes(*src, src_root_id, *dst, dst_root_id);
      if (!status.ok()) {
        return AttachDef(
            errors::InvalidArgument("Nodes were connected by a "
                                    "reference connection (requiring them to "
                                    "be on the same device), but the two nodes "
                                    "were assigned two different devices: ",
                                    status.error_message()),
            *dst);
      }
    }
  }

  // 3. For each node, assign a device based on the constraints in the
  // disjoint node set.
  std::vector<Node*> second_pass;
  for (Node* node : graph_->op_nodes()) {
    // The graph may have come pre-populated by the framework with assigned
    // devices (e.g., for stateful placements), so the placer should not try to
    // place nodes that are already placed.
    if (node->has_assigned_device_name()) {
      LogDeviceAssignment(node);
      continue;
    }

    // Heuristic A: prefer to place "generators" with their only
    // consumers.
    //
    // If this is a node with no inputs and one output, we save
    // this for a second pass, so that the consumer's placement
    // is chosen.
    if (IsGeneratorNode(node)) {
      second_pass.push_back(node);
      continue;
    }

    std::vector<Device*>* devices;
    Status status = colocation_graph.GetDevicesForNode(node, &devices);
    if (!status.ok()) {
      return AttachDef(
          errors::InvalidArgument("Cannot assign a device for operation '",
                                  node->name(), "': ", status.error_message()),
          *node);
    }

    // Returns the first device in sorted devices list so we will always
    // choose the same device.
    //
    // TODO(vrv): Factor this assignment out into a pluggable
    // algorithm, so that Placer is responsible for enforcing
    // preconditions and we can experiment with other algorithms when
    // given a choice of devices. Once we have a better idea of the
    // types of heuristics we want to use and the information needed
    // to perform good placement we can add an interface for this.
    int assigned_device = -1;

    // Heuristic B: If the node only operates on metadata, not data,
    // then it is desirable to place that metadata node with its
    // input.
    if (IsMetadata(node)) {
      // Make sure that the input device type is in the list of supported
      // device types for this node.
      const Node* input = (*node->in_edges().begin())->src();
      // TODO(vrv): if the input is empty, consider postponing this
      // node's assignment to the second pass, so that we handle the
      // case where a metadata node's input comes from a backedge
      // of a loop.
      if (CanAssignToDevice(input->assigned_device_name(), *devices)) {
        assigned_device = input->assigned_device_name_index();
      }
    }

    // Provide the default, if necessary.
    if (assigned_device == -1) {
      assigned_device = graph_->InternDeviceName((*devices)[0]->name());
    }

    AssignAndLog(assigned_device, node);
  }

  // 4. Perform a second pass assignment for those nodes explicitly
  // skipped during the first pass.
  for (Node* node : second_pass) {
    std::vector<Device*>* devices;
    Status status = colocation_graph.GetDevicesForNode(node, &devices);
    if (!status.ok()) {
      return AttachDef(
          errors::InvalidArgument("Cannot assign a device for operation '",
                                  node->name(), "': ", status.error_message()),
          *node);
    }

    int assigned_device = -1;

    // Heuristic A application.
    if (IsGeneratorNode(node)) {
      const Node* output = (*node->out_edges().begin())->dst();
      int output_device_name = output->assigned_device_name_index();

      const bool consumers_on_same_device = std::all_of(
          node->out_edges().begin(), node->out_edges().end(),
          [output_device_name](const Edge* e) {
            return e->dst()->assigned_device_name_index() == output_device_name;
          });

      if (consumers_on_same_device &&
          CanAssignToDevice(output->assigned_device_name(), *devices)) {
        assigned_device = output_device_name;
      }
    }

    // Provide the default, if necessary.
    if (assigned_device == -1) {
      assigned_device = graph_->InternDeviceName((*devices)[0]->name());
    }

    AssignAndLog(assigned_device, node);
  }

  return Status::OK();
}

bool Placer::CanAssignToDevice(const string& candidate_device_name,
                               const std::vector<Device*>& devices) const {
  if (!candidate_device_name.empty()) {
    // 'devices' lists the set of devices that the placer or the user has
    // constrained the operation to.  "candidate_device_name" must
    // refer to a concrete Device that is in the list of 'devices'.
    const Device* other_device =
        devices_->FindDeviceByName(candidate_device_name);
    if (std::find(devices.begin(), devices.end(), other_device) !=
        devices.end()) {
      return true;
    }
  }

  return false;
}

void Placer::AssignAndLog(int assigned_device, Node* node) const {
  node->set_assigned_device_name_index(assigned_device);
  LogDeviceAssignment(node);
}

void Placer::LogDeviceAssignment(const Node* node) const {
  // Log placement if log_device_placement is set.
  if (log_device_placement_) {
    printf("%s: (%s): %s\n", node->name().c_str(), node->type_string().c_str(),
           node->assigned_device_name().c_str());
    LOG(INFO) << node->name() << ": "
              << "(" << node->type_string() << ")"
              << node->assigned_device_name();
  }
}

}  // namespace tensorflow
