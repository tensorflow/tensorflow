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

#include "tensorflow/core/common_runtime/simple_placer.h"

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

namespace {

// Returns a list of devices sorted by preferred type and then name
// from 'devices' whose type is in 'supported_device_types'.  This
// function searches the device types in 'supported_device_types' and
// returns the subset of devices that match.
std::vector<Device*> FilterSupportedDevices(
    const std::vector<Device*>& devices,
    const DeviceTypeVector& supported_device_types) {
  std::vector<Device*> filtered_devices;
  for (DeviceType d : supported_device_types) {
    for (Device* device : devices) {
      if (DeviceType(device->attributes().device_type()) == d) {
        filtered_devices.emplace_back(device);
      }
    }
  }

  auto device_sort = [](const Device* a, const Device* b) {
    // First sort by prioritized device type and then by device name.
    return std::make_pair(
               DeviceSet::DeviceTypeOrder(DeviceType(a->device_type())),
               StringPiece(a->name())) <
           std::make_pair(
               DeviceSet::DeviceTypeOrder(DeviceType(b->device_type())),
               StringPiece(b->name()));
  };
  std::sort(filtered_devices.begin(), filtered_devices.end(), device_sort);
  return filtered_devices;
}

// Returns the name of the colocation group of the node by inspecting
// the "_class" attribute of the NodeDef.  Returns "" if it doesn't
// exist.
Status ColocationGroups(const Node& node,
                        std::vector<string>* colocation_groups) {
  std::vector<string> class_specs;
  // TODO(vrv): We should consider adding a GetNodeAttr that returns a
  // StringPiece, to avoid a copy.
  Status s = GetNodeAttr(node.def(), "_class", &class_specs);
  if (!s.ok()) {
    // No "_class" attribute is equivalent to the empty colocation_group.
    *colocation_groups = {strings::StrCat("loc:@", node.name())};
    return Status::OK();
  }

  bool found_spec = false;
  for (const string& class_spec : class_specs) {
    StringPiece spec(class_spec);
    if (spec.Consume("loc:@")) {
      found_spec = true;
      colocation_groups->emplace_back(class_spec);
    }
  }

  if (!found_spec) {
    *colocation_groups = {strings::StrCat("loc:@", node.name())};
  }
  return Status::OK();
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
                  const SessionOptions* options)
      : device_set_(device_set),
        device_types_(device_set->PrioritizedDeviceTypeList()),
        options_(options) {
    members_.reserve(graph->num_node_ids());
  }

  // Adds the given node to this ColocationGraph as a singleton.
  //
  // NOTE: The implementation assumes that the ids of nodes passed to
  // this method are dense and zero-based; the memory used will be linear in
  // the largest node ID.
  // NOTE: If this method returns an error, *this is left in an undefined
  // state.
  Status AddNode(const Node& node) {
    Member member;
    TF_RETURN_IF_ERROR(InitializeMember(node, &member));
    CHECK_GE(member.parent, 0);
    members_.resize(member.parent + 1);
    members_[member.parent] = std::move(member);

    // When adding the node, identify whether it is part of a
    // colocation group.
    std::vector<string> colocation_groups;
    TF_RETURN_IF_ERROR(ColocationGroups(node, &colocation_groups));
    Status s;
    for (const string& colocation_group : colocation_groups) {
      auto it = colocation_group_root_.find(colocation_group);
      if (it == colocation_group_root_.end()) {
        // This is the first node of the colocation group, so
        // designate this node as the 'root' of that colocation group.
        colocation_group_root_[colocation_group] = &node;
      } else {
        // Try to colocate the node with the root.  If there is an
        // error, return it.
        s = ColocateNodes(node, *(it->second));
        if (!s.ok()) {
          return s;
        }
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

    Status s;
    if (x_root != y_root) {
      // Merge the sets by swinging the parent pointer of the smaller
      // tree to point to the root of the larger tree. Together with
      // path compression in ColocationGraph::FindRoot, this ensures
      // that we do not experience pathological performance on graphs
      // such as chains.
      int new_root, old_root;
      if (members_[x_root].rank < members_[y_root].rank) {
        // The tree rooted at x_root is shallower, so connect it to
        // y_root. The rank of y_root is unchanged because its new
        // child has strictly less rank.
        members_[x_root].parent = y_root;
        new_root = y_root;
        old_root = x_root;
      } else if (members_[x_root].rank > members_[y_root].rank) {
        // The tree rooted at y_root is shallower, so connect it to
        // x_root. The rank of x_root is unchanged because its new
        // child has strictly less rank.
        members_[y_root].parent = x_root;
        new_root = x_root;
        old_root = y_root;
      } else {
        // Both trees have the same rank, so break the tie by choosing
        // x_root as the new root.
        members_[y_root].parent = x_root;
        // Increment the rank of the tree rooted at x_root, because it
        // is now strictly deeper than before.
        ++members_[x_root].rank;
        new_root = x_root;
        old_root = y_root;
      }

      // Merge the partial device specifications, and ensure that they are
      // compatible. NULL options_ is treated as allowing soft placement.
      // TODO(mrry): Consider enriching the error message by pointing
      // out which nodes have the explicit partial device
      // specifications that caused this conflict.
      s = DeviceNameUtils::MergeDevNames(
          &members_[new_root].device_name, members_[old_root].device_name,
          options_ == nullptr || options_->config.allow_soft_placement());
      if (!s.ok()) {
        return errors::InvalidArgument("Cannot colocate nodes '", x.name(),
                                       "' and '", y.name(), ": ",
                                       s.error_message());
      }

      // Transfer ids in the old group to the new one.
      members_[new_root].ids_in_group.insert(
          members_[old_root].ids_in_group.begin(),
          members_[old_root].ids_in_group.end());
      members_[old_root].ids_in_group.clear();

      // Ensure that the common root has at least one supported device
      // type, by computing the intersection of
      // members_[new_root].supported_device_types and
      // members_[old_root].supported_device_types.
      MergeSupportedDevices(&members_[new_root].supported_device_types,
                            members_[old_root].supported_device_types);
      if (members_[x_root].supported_device_types.size() == 0) {
        return errors::InvalidArgument(
            "Cannot colocate nodes '", x.name(), "' and '", y.name(),
            "' because no device type supports both of those nodes and the "
            "other nodes colocated with them");
      }
    }
    return Status::OK();
  }

  // Returns the device name associated with 'node'.
  DeviceNameUtils::ParsedName DeviceForNode(const Node& node) {
    int node_root = FindRoot(node.id());
    return members_[node_root].device_name;
  }

  void SetDeviceForNode(Node* node, const DeviceNameUtils::ParsedName& device) {
    int node_root = FindRoot(node->id());
    members_[node_root].device_name = device;
  }

  // For the given node, subject to the constraints previously given
  // to this ColocationGraph, set its assigned_device_name. Returns OK
  // if a satisfying device can be found, otherwise an error.
  Status GetDevicesForNode(Node* node, std::vector<Device*>* possible_devices) {
    possible_devices->clear();
    const int node_root = FindRoot(node->id());
    if (!members_[node_root].possible_devices.empty()) {
      *possible_devices = members_[node_root].possible_devices;
      return Status::OK();
    }

    // String containing additional debugging info on failures.
    string debug_info;

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

      // Perform soft placement if allow_soft_placement is set.  options_
      // being NULL is treated as allowing soft placement.
      if (devices.empty() &&
          (options_ == nullptr || options_->config.allow_soft_placement())) {
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
        AddDebugInfo(node_root, &debug_info);

        DeviceNameUtils::ParsedName specified_device_name;
        if (DeviceNameUtils::ParseFullName(node->def().device(),
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
                "Could not satisfy explicit device specification '",
                node->def().device(),
                "' because no devices matching that specification "
                "are registered in this process; available devices: ",
                str_util::Join(device_names, ", "), debug_info);
          } else if (specified_device_name.has_type) {
            return errors::InvalidArgument(
                "Could not satisfy explicit device specification '",
                node->def().device(), "' because no supported kernel for ",
                specified_device_name.type, " devices is available.",
                debug_info);
          } else {
            return errors::InvalidArgument(
                "Could not satisfy explicit device specification '",
                node->def().device(), debug_info);
          }
        } else {
          // The specified device may be a valid device but the
          // merged set device is different, so print both.
          return errors::InvalidArgument(
              "Could not satisfy explicit device specification '",
              node->def().device(),
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
        AddDebugInfo(node_root, &debug_info);
        return errors::InvalidArgument(
            "Node had no OpKernel registered to support this operation: ",
            "Operation was ", node->type_string(), " and inputs were ",
            DataTypeVectorString(node->input_types()), debug_info);
      }
    }

    // Cache the result of the possible devices for this node group.
    members_[node_root].possible_devices = devices;
    *possible_devices = members_[node_root].possible_devices;
    return Status::OK();
  }

 private:
  // Represents a node in the disjoint node set forest, and the
  // accumulated constraints on the device used by that node.
  struct Member {
    Member() = default;
    // The id of the node that is the parent of this one, or its own
    // id if it is a root. parent <= 0 indicates that this member is invalid.
    int parent = -1;

    // The set of ids that are part of the disjoint node set forest.
    //
    // This is only fully specified in the root of a disjoint
    // node set forest.
    std::set<int> ids_in_group;

    // The type of the op for this node.
    string op_type;

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

  // Adds debugging info to 'output' for the node referred to by
  // 'node_root'.
  void AddDebugInfo(const int node_root, string* output) {
    if (members_[node_root].ids_in_group.size() > 1) {
      strings::StrAppend(output, "\nColocation Debug Info:\n");

      // If this node is part of a colocation group, then we want to
      // collect the mapping of ops to supported devices, so that
      // the user can see why an unsatisfiable placement occurred.
      strings::StrAppend(
          output, "Colocation group had the following types and devices: ");

      std::unordered_map<string, string> type_to_devices;
      for (const int id : members_[node_root].ids_in_group) {
        const string& op_type = members_[id].op_type;
        string devices_registered;
        for (const auto& device_type : members_[id].supported_device_types) {
          strings::StrAppend(&devices_registered, DeviceTypeString(device_type),
                             " ");
        }

        type_to_devices[op_type] = devices_registered;
      }

      for (const auto& td : type_to_devices) {
        strings::StrAppend(output, "\n", td.first, ": ", td.second);
      }
    }
  }

  Status InitializeMember(const Node& node, Member* member) {
    const int id = node.id();
    member->ids_in_group.insert(id);
    member->op_type = node.type_string();

    if (id < 0) {
      return errors::InvalidArgument("Node id was not positive: ", id);
    }
    member->parent = id;
    TF_RETURN_IF_ERROR(SupportedDeviceTypesForNode(
        device_types_, node.def(), &member->supported_device_types));

    if (!node.assigned_device_name().empty()) {
      // This node has already been assigned to a device, so we
      // respect this placement, after sanity-checking it.  The
      // device_name and supported_device_types for this node reflect
      // the assigned device, so any nodes colocated with this node
      // will be assigned to the same device (assuming this is
      // possible).
      // NOTE: Since any assignment must have been performed by
      // the TensorFlow runtime, we consider errors in this branch to
      // be INTERNAL.
      if (!DeviceNameUtils::ParseFullName(node.assigned_device_name(),
                                          &member->device_name)) {
        return errors::Internal("Malformed assigned device '",
                                node.assigned_device_name(), "'");
      }
      std::vector<Device*> devices;
      const Device* assigned_device =
          device_set_->FindDeviceByName(node.assigned_device_name());
      if (assigned_device == nullptr) {
        return errors::Internal("Assigned device '",
                                node.assigned_device_name(),
                                "' does not match any device");
      }

      for (DeviceType d : member->supported_device_types) {
        if (DeviceType(assigned_device->attributes().device_type()) == d) {
          return Status::OK();
        }
      }

      return errors::Internal("Assigned device '", node.assigned_device_name(),
                              "' does not have registered OpKernel support "
                              "for ",
                              node.def().op());
    } else {
      // This node has not yet been assigned to a device, so we
      // calculate any constraints due to the set of registered
      // kernels and any (partial) user-provided device specification
      // in the NodeDef.

      // If no kernels are registered for this op type, fail with an error.
      if (member->supported_device_types.empty()) {
        return errors::InvalidArgument(
            "No OpKernel was registered to support "
            "Op '",
            node.def().op(), "' with these attrs");
      }

      // If the NodeDef contains a device, then we interpret it as a
      // (partial) device specification.
      if (!node.def().device().empty()) {
        // The user has specified a device in the NodeDef, try to find a
        // valid device matching their specification in the set of
        // devices.
        // NOTE: The full name may specify a device that is not in
        // n.supported_device_types(), but we check that in AssignDevice().
        if (!DeviceNameUtils::ParseFullName(node.def().device(),
                                            &member->device_name)) {
          return errors::InvalidArgument("Malformed device specification '",
                                         node.def().device(), "'");
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
    for (DeviceType device_type : temp) {
      bool found = false;
      for (DeviceType other_device_type : other) {
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
    DCHECK_GE(members_[node_id].parent, 0);
    if (members_[node_id].parent != node_id) {
      // NOTE: Compress paths from node_id to its root, so that future
      // calls to FindRoot and ColocateNodes are more efficient.
      members_[node_id].parent = FindRoot(members_[node_id].parent);
    }
    return members_[node_id].parent;
  }

  std::vector<Member> members_;
  const DeviceSet* device_set_;  // Not owned.
  const std::vector<DeviceType> device_types_;
  const SessionOptions* options_;  // Not owned;

  // Maps from a colocation group identifier to the 'root' of that
  // colocation group.
  std::unordered_map<string, const Node*> colocation_group_root_;
};

// Returns true if the node only depends on its input's metadata
// (shape).  Not necessarily a complete list.
bool IsMetadataNode(const Node* node) {
  const string& node_type = node->type_string();
  return (node_type == "Size" || node_type == "Shape" || node_type == "Rank");
}

// Returns true if the node has no inputs and produces outputs
// that are consumed by a single node.
//
// TODO(vrv): Currently this handles only nodes with one output, but
// this could be extended to handle the case where a node has many
// outputs that are connected to nodes in the same colocation group.
bool IsGeneratorNode(const Node* node) {
  return node->num_inputs() == 0 && node->num_outputs() == 1 &&
         node->out_edges().size() == 1 && !IsRefType(node->output_type(0));
}

}  // namespace

SimplePlacer::SimplePlacer(Graph* graph, const DeviceSet* devices,
                           const SessionOptions* options)
    : graph_(graph),
      devices_(devices),
      options_(options) {}

SimplePlacer::SimplePlacer(Graph* graph, const DeviceSet* devices)
    : graph_(graph), devices_(devices) {
  options_ = nullptr;
}

SimplePlacer::~SimplePlacer() {}

Status SimplePlacer::Run() {
  if (devices_->devices().empty()) {
    return errors::FailedPrecondition("No devices are registered");
  }

  ColocationGraph colocation_graph(graph_, devices_, options_);
  Status status;

  // 1. First add all of the nodes. Note that steps (1) and (2)
  // requires two passes over the nodes because the graph (and hence
  // the constraints) may not be acyclic.
  for (Node* node : graph_->nodes()) {
    // Skip the source and sink nodes.
    if (!node->IsOp()) {
      continue;
    }
    status = colocation_graph.AddNode(*node);
    if (!status.ok()) return AttachDef(status, node->def());
  }

  // 2. Enumerate the constraint edges, and use them to update the disjoint
  // node set.
  for (Node* node : graph_->nodes()) {
    if (!node->IsOp()) {
      continue;
    }

    // If `node` has an input edge with reference type, add an
    // edge from the source of that edge to `node`.
    for (const auto& edge : node->in_edges()) {
      if (!edge->IsControlEdge() &&
          IsRefType(node->input_type(edge->dst_input()))) {
        // If both the source node and this node have paritally
        // specified a device, then 'node's device should be
        // cleared: the reference edge forces 'node' to be on the
        // same device as the source node.
        auto source_parsed_name = colocation_graph.DeviceForNode(*edge->src());
        auto dest_parsed_name = colocation_graph.DeviceForNode(*node);
        if (DeviceNameUtils::HasSomeDetails(source_parsed_name) &&
            DeviceNameUtils::HasSomeDetails(dest_parsed_name)) {
          // Add a log saying that we are ignoring a specified device
          // for 'node' if the two names were incompatible.
          if (!DeviceNameUtils::AreCompatibleDevNames(source_parsed_name,
                                                      dest_parsed_name)) {
            LOG(INFO) << "Ignoring device specification "
                      << DeviceNameUtils::ParsedNameToString(
                             colocation_graph.DeviceForNode(*node))
                      << " for node '" << node->name()
                      << "' because the input edge from '"
                      << edge->src()->name()
                      << "' is a reference connection and already has a device "
                         "field set to "
                      << DeviceNameUtils::ParsedNameToString(
                             colocation_graph.DeviceForNode(*edge->src()));

            // Make 'node' colocated with the source
            colocation_graph.SetDeviceForNode(node, source_parsed_name);
          } else {
            bool source_subset_of_dest = DeviceNameUtils::IsSpecification(
                source_parsed_name, dest_parsed_name);
            bool dest_subset_of_source = DeviceNameUtils::IsSpecification(
                dest_parsed_name, source_parsed_name);

            if (source_subset_of_dest && !dest_subset_of_source) {
              colocation_graph.SetDeviceForNode(edge->src(), dest_parsed_name);
            } else {
              colocation_graph.SetDeviceForNode(node, source_parsed_name);
            }
          }
        }

        status = colocation_graph.ColocateNodes(*edge->src(), *node);
        if (!status.ok()) {
          return AttachDef(errors::InvalidArgument(
                               "Nodes were connected by a "
                               "reference connection (requiring them to "
                               "be on the same device), but the two nodes "
                               "were assigned two different devices: ",
                               status.error_message()),
                           node->def());
        }
      }
    }
  }

  // 3. For each node, assign a device based on the constraints in the
  // disjoint node set.
  std::vector<Device*> devices;
  std::vector<Node*> second_pass;
  for (Node* node : graph_->nodes()) {
    // Skip the source and sink nodes.
    if (!node->IsOp()) {
      continue;
    }
    // Skip nodes that already have an assigned name.
    if (!node->assigned_device_name().empty()) {
      continue;
    }

    // Heuristic A: prefer to place "generators" with their only
    // consumers.
    //
    // If this is a node with no inputs and a single (non-ref)
    // consumer, we save this for a second pass, so that the
    // consumer's placement is chosen.
    if (IsGeneratorNode(node)) {
      second_pass.push_back(node);
      continue;
    }

    status = colocation_graph.GetDevicesForNode(node, &devices);
    if (!status.ok()) {
      return AttachDef(
          errors::InvalidArgument("Cannot assign a device to node '",
                                  node->name(), "': ", status.error_message()),
          node->def());
    }

    // Returns the first device in sorted devices list so we will always
    // choose the same device.
    //
    // TODO(vrv): Factor this assignment out into a pluggable
    // algorithm, so that SimplePlacer is responsible for enforcing
    // preconditions and we can experiment with other algorithms when
    // given a choice of devices. Once we have a better idea of the
    // types of heuristics we want to use and the information needed
    // to perform good placement we can add an interface for this.
    string assigned_device = devices[0]->name();

    // Heuristic B: If the node only operates on metadata, not data,
    // then it is desirable to place that metadata node with its
    // input.
    if (IsMetadataNode(node)) {
      // Make sure that the input device type is in the list of supported
      // device types for this node.
      const Node* input = (*node->in_edges().begin())->src();
      // TODO(vrv): if the input is empty, consider postponing this
      // node's assignment to the second pass, so that we handle the
      // case where a metadata node's input comes from a backedge
      // of a loop.
      const string& input_device_name = input->assigned_device_name();
      if (CanAssignToDevice(input_device_name, devices)) {
        assigned_device = input_device_name;
      }
    }

    AssignAndLog(assigned_device, node);
  }

  // 4. Perform a second pass assignment for those nodes explicitly
  // skipped during the first pass.
  for (Node* node : second_pass) {
    status = colocation_graph.GetDevicesForNode(node, &devices);
    if (!status.ok()) {
      return AttachDef(
          errors::InvalidArgument("Cannot assign a device to node '",
                                  node->name(), "': ", status.error_message()),
          node->def());
    }

    string assigned_device = devices[0]->name();

    // Heuristic A application.
    if (IsGeneratorNode(node)) {
      const Node* output = (*node->out_edges().begin())->dst();
      const string& output_device_name = output->assigned_device_name();
      if (CanAssignToDevice(output_device_name, devices)) {
        assigned_device = output_device_name;
      }
    }

    AssignAndLog(assigned_device, node);
  }

  return Status::OK();
}

bool SimplePlacer::CanAssignToDevice(const string& candidate_device_name,
                                     const std::vector<Device*> devices) const {
  if (!candidate_device_name.empty()) {
    // Can we assign to the same device?  Check by validating that
    // the device type of 'candidate_device_name' is present
    // in 'devices'.
    const Device* other_device =
        devices_->FindDeviceByName(candidate_device_name);
    if (std::any_of(devices.begin(), devices.end(), [other_device](Device* d) {
          return d->device_type() == other_device->device_type();
        })) {
      return true;
    }
  }

  return false;
}

void SimplePlacer::AssignAndLog(const string& assigned_device,
                                Node* node) const {
  node->set_assigned_device_name(assigned_device);
  // Log placement if log_device_placement is set.
  if (options_ && options_->config.log_device_placement()) {
    printf("%s: %s\n", node->name().c_str(),
           node->assigned_device_name().c_str());
    LOG(INFO) << node->name() << ": " << node->assigned_device_name();
  }
}

}  // namespace tensorflow
