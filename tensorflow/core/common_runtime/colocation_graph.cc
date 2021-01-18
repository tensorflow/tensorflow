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

#include "tensorflow/core/common_runtime/colocation_graph.h"

#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/composite_device.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/common_runtime/inspecting_placer.h"
#include "tensorflow/core/common_runtime/partitioning_utils.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/port.h"

namespace tensorflow {

namespace {

// We hoist the conversion from C-style string literal to StringPiece here,
// so that we can avoid the many repeated calls to strlen().
const StringPiece kColocationAttrNameStringPiece(kColocationAttrName);
const StringPiece kColocationGroupPrefixStringPiece(kColocationGroupPrefix);

// Using absl::StrJoin with lambda does not work in tf-lite builds.
std::vector<string> DevicesToString(const std::vector<Device*> devices) {
  std::vector<string> v;
  v.reserve(devices.size());
  for (Device* d : devices) {
    v.push_back(d->name());
  }
  return v;
}

// Using absl::StrJoin with lambda does not work in tf-lite builds.
std::vector<string> DeviceTypeAndPriorityToString(
    const PrioritizedDeviceTypeVector& devices) {
  std::vector<string> v;
  v.reserve(devices.size());
  for (const std::pair<DeviceType, int32>& device_and_type : devices) {
    v.push_back(DeviceTypeString(device_and_type.first));
  }
  return v;
}

bool IsRefOrResource(DataType data_type) {
  return IsRefType(data_type) || data_type == DT_RESOURCE;
}

// While Placer can override requested device on ops processing
// resources, i.e. node that take (and potentially return) a resource,
// it must not override requested device on ops generating a resource,
// e.g. VarHandleOp, _Arg. Such ops are currently no-input, single resource/ref
// output nodes.
bool IsRefOrResourceGeneratorNode(const Node& node) {
  return node.num_inputs() == 0 && node.num_outputs() == 1 &&
         IsRefOrResource(node.output_type(0));
}

bool IsExemptFromResourceInputColocation(const Node* node) {
  // Note: Partitioned function calls, which place and partition their
  // function bodies, are exempt from this check: they forward resource and
  // ref inputs to operations that are appropriately placed, instead of
  // dereferencing them.
  const string& op_type = node->op_def().name();
  auto exempt_ops = InputColocationExemptionRegistry::Global()->Get();
  return exempt_ops.find(op_type) != exempt_ops.end();
}

bool HasPriorities(const PrioritizedDeviceTypeVector& device_types) {
  for (const auto& prioritized_device_type : device_types) {
    if (prioritized_device_type.second != 0) return true;
  }
  return false;
}

bool ArePrioritiesSame(const PrioritizedDeviceTypeVector& a_types,
                       const PrioritizedDeviceTypeVector& b_types) {
  if (a_types.size() != b_types.size()) {
    return false;
  }
  for (int i = 0; i < a_types.size(); ++i) {
    if (a_types[i].first != b_types[i].first) {
      return false;
    }
  }
  return true;
}

bool IsXlaDevice(absl::string_view device_type) {
  if (device_type == "XLA_CPU_JIT" || device_type == "XLA_GPU_JIT" ||
      device_type == "XLA_TPU_JIT") {
    // Symbolic XLA device.
    return true;
  }

  return (device_type == "XLA_CPU" || device_type == "XLA_GPU" ||
          device_type == "TPU");
}

bool IsCompositeDevice(absl::string_view device_type) {
  return device_type == kCompositeDeviceType;
}

}  // namespace

Status Member::SetParentAndSupportedDevices(
    const Node& node, const std::vector<DeviceType>& types,
    const DeviceNameUtils::ParsedName* local_address_spec) {
  int id = node.id();
  if (id < 0) {
    return errors::Internal("Placer should not be creating a Member for node: ",
                            node.DebugString());
  }
  parent_ = id;
  return SupportedDeviceTypesForNode(
      types, node.def(), &supported_device_types_, local_address_spec);
}

Status Member::SetAssignedDeviceName(const string& device_name) {
  if (DeviceNameUtils::HasSomeDetails(requested_device_name_)) {
    return errors::Internal(
        "Setting assigned device name when there is a requested device set "
        "is unsupported");
  }
  if (!DeviceNameUtils::ParseFullName(device_name, &assigned_device_name_)) {
    return errors::Internal("Malformed assigned device '", device_name, "'");
  }
  // Set requested device to assigned_device to maintain the invariant that
  // requested is a specialization of assigned.
  requested_device_name_ = assigned_device_name_;
  return Status::OK();
}

Status Member::SetResourceDeviceName(const Node& node) {
  if (DeviceNameUtils::HasSomeDetails(requested_device_name_)) {
    return errors::Internal(
        "Setting resource device name when there is a requested device set "
        "is unsupported");
  }

  if (!DeviceNameUtils::ParseFullName(node.requested_device(),
                                      &resource_device_name_)) {
    return errors::InvalidArgument("Malformed device specification '",
                                   node.requested_device(),
                                   "' in node: ", node.DebugString());
  }

  // Set requested device to resource device to maintain the invariant that
  // requested is a specialization of resource.
  requested_device_name_ = resource_device_name_;
  return Status::OK();
}

Status Member::SetRequestedDeviceName(const Node& node) {
  if (DeviceNameUtils::HasSomeDetails(assigned_device_name_)) {
    return errors::Internal(
        "Setting requested device name when there is an assigned device set "
        "is unsupported");
  }
  if (DeviceNameUtils::HasSomeDetails(resource_device_name_)) {
    return errors::Internal(
        "Setting requested device name when there is a resource device set "
        "is unsupported");
  }
  if (!DeviceNameUtils::ParseFullName(node.requested_device(),
                                      &requested_device_name_)) {
    return errors::InvalidArgument("Malformed device specification '",
                                   node.requested_device(),
                                   "' in node: ", node.DebugString());
  }
  return Status::OK();
}

Status Member::FillPossibleDevices(PossibleDevices* possible_device) const {
  if (DeviceNameUtils::HasSomeDetails(assigned_device_name_)) {
    return errors::Internal(
        "Cannot fill PossibleDevices from a member that has non-empty assigned "
        "device. Did we start assigning devices to functions called by deep "
        "ops? ",
        DebugString());
  }
  possible_device->requested_device_name = requested_device_name_;
  possible_device->resource_device_name = resource_device_name_;
  possible_device->device_types = supported_device_types_;
  return Status::OK();
}

bool Member::IsEdgeFromCompositeDeviceToPhysicalDevice(
    const Member& src_root) const {
  auto compatible_edge_from_composite_device_to_physical_device =
      [](const DeviceNameUtils::ParsedName& src_device,
         const DeviceNameUtils::ParsedName& dst_device) -> bool {
    return src_device.has_type && dst_device.has_type &&
           IsCompositeDevice(src_device.type) &&
           !IsCompositeDevice(dst_device.type);
  };
  if (compatible_edge_from_composite_device_to_physical_device(
          src_root.assigned_device_name_, assigned_device_name_) ||
      compatible_edge_from_composite_device_to_physical_device(
          src_root.resource_device_name_, resource_device_name_) ||
      compatible_edge_from_composite_device_to_physical_device(
          src_root.requested_device_name_, requested_device_name_)) {
    return true;
  }
  return false;
}

Status Member::EnsureCompatibilityAcrossResourceEdge(
    const Node& src, const Member& src_root,
    const Node& dst, /*dst_root is this*/
    bool log_device_placement) {
  if (!DeviceNameUtils::AreCompatibleDevNames(src_root.assigned_device_name_,
                                              assigned_device_name_)) {
    return errors::InvalidArgument(
        "Cannot place the graph because a reference or resource edge "
        "connects colocation groups with incompatible assigned devices: ",
        DeviceNameUtils::ParsedNameToString(src_root.assigned_device_name_),
        " vs ", DeviceNameUtils::ParsedNameToString(assigned_device_name_),
        ". The edge src node is ", src.name(), " , and the dst node is ",
        dst.name());
  }

  if (!DeviceNameUtils::AreCompatibleDevNames(src_root.resource_device_name_,
                                              resource_device_name_)) {
    return errors::InvalidArgument(
        "Cannot place the graph because a reference or resource edge "
        "connects colocation groups with incompatible resource devices: ",
        DeviceNameUtils::ParsedNameToString(src_root.resource_device_name_),
        " vs ", DeviceNameUtils::ParsedNameToString(resource_device_name_),
        ". The edge src node is ", src.name(), " , and the dst node is ",
        dst.name());
  }

  if (DeviceNameUtils::AreCompatibleDevNames(src_root.requested_device_name_,
                                             requested_device_name_)) {
    return Status::OK();
  }

  // If we are here, assigned and resource devices are compatible but requested
  // ones are not. We will be overriding the requested device for destination
  // node, but need to preserve the invariant that it will be a specialization
  // of the assigned and resource devices.
  if (log_device_placement) {
    LOG(INFO) << "Ignoring device specification "
              << DeviceNameUtils::ParsedNameToString(requested_device_name_)
              << " for node '" << dst.name()
              << "' because the input edge from '" << src.name()
              << "' is a reference connection and already has a device "
                 "field set to "
              << DeviceNameUtils::ParsedNameToString(
                     src_root.requested_device_name_);
  }
  requested_device_name_ = src_root.requested_device_name_;
  DeviceNameUtils::EnsureSpecification(&requested_device_name_,
                                       assigned_device_name_);
  DeviceNameUtils::EnsureSpecification(&requested_device_name_,
                                       resource_device_name_);
  return Status::OK();
}

void Member::Merge(std::vector<Member>* tree, int x_root, int y_root,
                   Member** new_root, Member** old_root, bool dry_run) {
  Member& x_root_member = (*tree)[x_root];
  Member& y_root_member = (*tree)[y_root];

  // Merge the sets by setting the parent pointer of the smaller tree's root
  // node to point to the root of the larger tree. Together with path
  // compression in ColocationGraph::FindRoot, this ensures that we do not
  // experience pathological performance on graphs such as chains.
  int new_root_id, old_root_id;
  if (x_root_member.rank_ < y_root_member.rank_) {
    // The tree rooted at x_root is shallower, so connect it to
    // y_root. The rank of y_root is unchanged because its new
    // child has strictly less rank.
    if (!dry_run) {
      x_root_member.parent_ = y_root;
    }
    new_root_id = y_root;
    old_root_id = x_root;
  } else if (x_root_member.rank_ > y_root_member.rank_) {
    // The tree rooted at y_root is shallower, so connect it to
    // x_root. The rank of x_root is unchanged because its new
    // child has strictly less rank.
    if (!dry_run) {
      y_root_member.parent_ = x_root;
    }
    new_root_id = x_root;
    old_root_id = y_root;
  } else {
    if (!dry_run) {
      // Both trees have the same rank, so break the tie by choosing
      // x_root as the new root.
      y_root_member.parent_ = x_root;
      // Increment the rank of the tree rooted at x_root, because it
      // is now strictly deeper than before.
      ++x_root_member.rank_;
    }
    new_root_id = x_root;
    old_root_id = y_root;
  }

  *new_root = &(*tree)[new_root_id];
  *old_root = &(*tree)[old_root_id];
}

// tree is non-const because we can change some `parent` pointers in some
// members for more efficient future lookups. The vector itself is not
// changed.
int Member::FindAndUpdateRoot(std::vector<Member>* tree, int node_id) {
  Member& member = (*tree)[node_id];
  if (member.parent_ == node_id) {
    // member.parent is the root of this disjoint tree.  Do nothing.
  } else {
    member.parent_ = FindAndUpdateRoot(tree, member.parent_);
  }
  // Now it is guaranteed that member.parent is the root of this disjoint
  // tree.
  return member.parent_;
}

int Member::FindRoot(const std::vector<Member>& tree, int node_id) {
  const Member& member = tree[node_id];
  if (member.parent_ == node_id) {
    return member.parent_;
  }
  return FindRoot(tree, member.parent_);
}

Status Member::MergeDeviceNames(const Member& other,
                                bool allow_soft_placement) {
  // Assuming the "requested is a specialization of assigned and resource
  // devices" invariant holds for this and `other`, it will hold after the
  // merges below.
  DeviceNameUtils::ParsedName assigned_device_name_copy = assigned_device_name_;
  TF_RETURN_IF_ERROR(DeviceNameUtils::MergeDevNames(
      &assigned_device_name_copy, other.assigned_device_name_));

  DeviceNameUtils::ParsedName resource_device_name_copy = resource_device_name_;
  TF_RETURN_IF_ERROR(DeviceNameUtils::MergeDevNames(
      &resource_device_name_copy, other.resource_device_name_));

  DeviceNameUtils::ParsedName requested_device_name_copy =
      requested_device_name_;
  TF_RETURN_IF_ERROR(DeviceNameUtils::MergeDevNames(
      &requested_device_name_copy, other.requested_device_name_,
      allow_soft_placement));

  DeviceNameUtils::EnsureSpecification(&requested_device_name_copy,
                                       assigned_device_name_copy);
  DeviceNameUtils::EnsureSpecification(&requested_device_name_copy,
                                       resource_device_name_copy);

  // We checked for all errors, now change the devices.
  assigned_device_name_ = assigned_device_name_copy;
  resource_device_name_ = resource_device_name_copy;
  requested_device_name_ = requested_device_name_copy;
  return Status::OK();
}

// Updates this to contain the intersection of the device types in
// this and "other".
bool Member::MergeSupportedDevices(const Member& other) {
  return MergeSupportedDevices(other.supported_device_types_);
}

bool Member::MergeSupportedDevices(
    const PrioritizedDeviceTypeVector& other_devices) {
  // Generate intersection with priorities.
  // Each vector contains the same device types but with different priorities.
  // The priorities are taken from the corresponding source vector.
  PrioritizedDeviceTypeVector target_intersection;
  PrioritizedDeviceTypeVector other_intersection;

  for (const auto& prioritized_device_type : supported_device_types_) {
    bool found = false;
    for (const auto& other_prioritized_device_type : other_devices) {
      if (prioritized_device_type.first ==
          other_prioritized_device_type.first) {
        found = true;
        other_intersection.push_back(other_prioritized_device_type);
        break;
      }
    }
    if (found) {
      target_intersection.push_back(prioritized_device_type);
    }
  }

  DeviceSet::SortPrioritizedDeviceTypeVector(&target_intersection);
  DeviceSet::SortPrioritizedDeviceTypeVector(&other_intersection);

  PrioritizedDeviceTypeVector result;

  bool is_target_prioritized = HasPriorities(target_intersection);
  bool is_other_prioritized = HasPriorities(other_intersection);
  if (!is_target_prioritized && !is_other_prioritized) {
    // If neither are prioritized then we just return the original i.e. target
    // prioritization.
    result = target_intersection;
  } else if (is_target_prioritized && !is_other_prioritized) {
    // If only one is prioritized, then we respect priorities of that in the
    // intersection.
    result = target_intersection;
  } else if (!is_target_prioritized && is_other_prioritized) {
    result = other_intersection;
  } else {
    // If both have priorities and agree then we go with that. If the
    // prioritization order is different, then we just fallback to the default
    // i.e. what the DeviceTypeOrder suggests. In that case, we also set the
    // merged priorities to 0, so that downstream merges work correctly as well.
    if (ArePrioritiesSame(target_intersection, other_intersection)) {
      result = target_intersection;
    } else {
      for (const auto& prioritized_device : target_intersection) {
        result.push_back(std::make_pair(prioritized_device.first, 0));
      }
      DeviceSet::SortPrioritizedDeviceTypeVector(&result);
    }
  }

  if (result.empty()) {
    return false;
  }
  supported_device_types_ = result;
  return true;
}

Status Member::AssignDevice(const Node& node) {
  if (node.assigned_device_name_index() == assigned_device_name_index_) {
    return Status::OK();
  }

  DeviceNameUtils::ParsedName parsed;
  DeviceNameUtils::ParseFullName(node.assigned_device_name(), &parsed);
  Status s = DeviceNameUtils::MergeDevNames(&assigned_device_name_, parsed);
  if (!s.ok()) {
    return errors::Internal(
        "Constraining by assigned device should not cause an error. Original "
        "root's assigned device name: ",
        DeviceNameUtils::ParsedNameToString(assigned_device_name_),
        " node's assigned device name \"", node.assigned_device_name(),
        ". Error: ", s.error_message());
  }
  s = DeviceNameUtils::MergeOverrideDevNames(&resource_device_name_, parsed);
  if (!s.ok()) {
    return errors::Internal(
        "Constraining by assigned device should not cause an error. Original "
        "root's resource device name: ",
        DeviceNameUtils::ParsedNameToString(resource_device_name_),
        " node's assigned device name \"", node.assigned_device_name(),
        ". Error: ", s.error_message());
  }
  s = DeviceNameUtils::MergeOverrideDevNames(&requested_device_name_, parsed);
  if (!s.ok()) {
    return errors::Internal(
        "Constraining by assigned device should not cause an error. Original "
        "root's requested device name: \"",
        DeviceNameUtils::ParsedNameToString(requested_device_name_),
        "\", node's assigned device name \"", node.assigned_device_name(),
        "\". Error: ", s.error_message());
  }

  assigned_device_name_index_ = node.assigned_device_name_index();
  // Clear cached possible_devices, if any.
  possible_devices_.clear();
  return Status::OK();
}

void Member::MaybeExcludeXlaDevices() {
  for (const auto& parsed_name :
       {requested_device_name_, assigned_device_name_, resource_device_name_}) {
    // Don't exculde XLA devices from supported devices if member is explicitly
    // assigned to a CompositeDevice.
    if (parsed_name.has_type && (IsXlaDevice(parsed_name.type) ||
                                 IsCompositeDevice(parsed_name.type))) {
      return;
    }
  }

  PrioritizedDeviceTypeVector non_xla_types;
  absl::c_copy_if(supported_device_types_, std::back_inserter(non_xla_types),
                  [&](const std::pair<DeviceType, int32>& entry) {
                    return !IsXlaDevice(entry.first.type_string());
                  });

  // TODO(b/141216278) Remove all XLA device types from the supported device
  // types if the node has no requested/assigned/resource XLA device.
  if (!non_xla_types.empty() &&
      non_xla_types.size() < supported_device_types_.size()) {
    supported_device_types_ = std::move(non_xla_types);
  }
}

Status Member::LimitToPossibleDevices(const PossibleDevices& devices,
                                      bool allow_soft_placement) {
  TF_RETURN_IF_ERROR(DeviceNameUtils::MergeDevNames(
      &requested_device_name_, devices.requested_device_name,
      allow_soft_placement));
  TF_RETURN_IF_ERROR(DeviceNameUtils::MergeDevNames(
      &resource_device_name_, devices.resource_device_name));
  MergeSupportedDevices(devices.device_types);
  return Status::OK();
}

string Member::DebugString() const {
  return absl::StrCat(
      "Member(assigned_device_name_index_=", assigned_device_name_index_,
      " requested_device_name_='",
      DeviceNameUtils::ParsedNameToString(requested_device_name_),
      "' assigned_device_name_='",
      DeviceNameUtils::ParsedNameToString(assigned_device_name_),
      "' resource_device_name_='",
      DeviceNameUtils::ParsedNameToString(resource_device_name_),
      "' supported_device_types_=[",
      absl::StrJoin(DeviceTypeAndPriorityToString(supported_device_types_),
                    ", "),
      "] possible_devices_=[",
      absl::StrJoin(DevicesToString(possible_devices_), ", "), "]");
}

DeviceNameUtils::ParsedName Member::GetSoftDeviceName() const {
  DeviceNameUtils::ParsedName soft_device_name = requested_device_name_;
  if (!assigned_device_name_.has_type) {
    soft_device_name.type.clear();
    soft_device_name.has_type = false;
  }
  if (!assigned_device_name_.has_id) {
    soft_device_name.has_id = false;
  }
  return soft_device_name;
}

DeviceNameUtils::ParsedName Member::GetPreferredSoftDeviceName() const {
  DeviceNameUtils::ParsedName soft_device_name = requested_device_name_;
  if (!assigned_device_name_.has_type && !resource_device_name_.has_type) {
    soft_device_name.type.clear();
    soft_device_name.has_type = false;
  }
  if (!assigned_device_name_.has_id && !resource_device_name_.has_id) {
    soft_device_name.has_id = false;
  }
  return soft_device_name;
}

// Returns ParsedName whose address space (i.e. job, replica, task) identifies
// the address space directly accessible by the local process. If the address
// space is fully specified and it is exactly the same as the address space
// of a device, then all kernels of that device should be registered in the
// local process.
static const DeviceNameUtils::ParsedName LocalAddressSpec(
    const Device* client_device, const Device* default_local_device) {
  if (client_device != nullptr) {
    return DeviceNameUtils::AddressSpace(client_device->parsed_name());
  }

  if (default_local_device != nullptr) {
    return DeviceNameUtils::AddressSpace(default_local_device->parsed_name());
  }

  // TODO(b/139617593) Return the name of the first local device in device_set_
  // once we can trust the output of Device::IsLocal().
  return DeviceNameUtils::ParsedName();
}

ColocationGraph::ColocationGraph(const Graph* graph, const FunctionStack& stack,
                                 const FunctionLibraryDefinition* flib_def,
                                 const DeviceSet* device_set,
                                 const Device* default_local_device,
                                 bool allow_soft_placement,
                                 bool log_device_placement)
    : graph_(*graph),
      stack_(stack),
      flib_def_(*flib_def),
      inspecting_placer_(stack, flib_def, device_set, default_local_device,
                         allow_soft_placement, log_device_placement),
      inspection_required_checker_(graph, flib_def),
      device_set_(*device_set),
      device_types_(device_set->PrioritizedDeviceTypeList()),
      local_address_spec_(
          LocalAddressSpec(device_set->client_device(), default_local_device)),
      default_local_device_(default_local_device),
      allow_soft_placement_(allow_soft_placement),
      log_device_placement_(log_device_placement) {
  members_.resize(graph_.num_node_ids());
}

// Adds each node of the Graph to this ColocationGraph as a singleton.
//
// NOTE: The implementation assumes that the ids of nodes passed to
// this method are dense and zero-based; the memory used will be linear in
// the largest node ID.
// NOTE: If this method returns an error, *this is left in an undefined
// state.
Status ColocationGraph::ColocateAllNodes() {
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

  for (const Node* node : graph_.op_nodes()) {
    // When adding the node, identify whether it is part of a colocation
    // group.

    // This code is effectively the equivalent of GetNodeAttr() for a string
    // array, but it avoids all internal allocations (the allocation of the
    // backing store of the std::vector<string> as well as the copies of the
    // strings within it).  Instead, we combine the query of the colocation
    // attribute with the calls to ColocateNodeToGroup.
    const AttrValue* attr_value =
        node->attrs().Find(kColocationAttrNameStringPiece);
    if (attr_value != nullptr) {
      if (attr_value->has_list()) {
        for (const string& class_spec : attr_value->list().s()) {
          StringPiece spec(class_spec);
          if (absl::ConsumePrefix(&spec, kColocationGroupPrefixStringPiece)) {
            TF_RETURN_IF_ERROR(
                ColocateNodeToGroup(&colocation_group_root, node, spec));
          }
        }
      } else if (!attr_value->s().empty()) {
        LOG(ERROR) << "The value for colocation attribute '_class' must be a "
                      "list of strings, not a single string: "
                   << node->DebugString();
      }
    }

    // Each node belongs to a colocation group with the node's name.
    TF_RETURN_IF_ERROR(
        ColocateNodeToGroup(&colocation_group_root, node, node->name()));
  }

  return Status::OK();
}

Status ColocationGraph::ColocateResourceOrRefEdge(const Node* src,
                                                  const Node* dst) {
  // Colocate `src` and `dst` to maintain the invariant that nodes
  // connected by reference edges are colocated.
  int src_root_id = FindAndUpdateRoot(src->id());
  int dst_root_id = FindAndUpdateRoot(dst->id());
  auto& src_root = members_[src_root_id];
  auto& dst_root = members_[dst_root_id];

  if (dst_root.IsEdgeFromCompositeDeviceToPhysicalDevice(src_root)) {
    // If the src root is assigned to a composite device and the dst root is
    // assigned to a physical device, don't colocate the dst root with the src
    // root.
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(dst_root.EnsureCompatibilityAcrossResourceEdge(
      *src, src_root, *dst, log_device_placement_));
  Status status = ColocateNodes(*src, src_root_id, *dst, dst_root_id);
  if (!status.ok()) {
    return AttachDef(
        errors::InvalidArgument(
            "Nodes were connected by a reference or resource connection "
            "(requiring them to be on the same device), but the two nodes "
            "were assigned two different devices: ",
            status.error_message()),
        *dst);
  }
  return Status::OK();
}

Status ColocationGraph::ColocateResourceAndRefEdges(
    std::unordered_set<Node*>* inspection_required) {
  // If `node` has an input edge with reference type, add an edge from the
  // source of that edge to `node`.
  for (const Edge* edge : graph_.edges()) {
    if (edge->IsControlEdge()) {
      continue;
    }
    Node* src = edge->src();
    Node* dst = edge->dst();
    bool needs_inspection;
    TF_RETURN_IF_ERROR(inspection_required_checker_.IsPlacerInspectionRequired(
        *src, &needs_inspection));
    if (needs_inspection) {
      inspection_required->insert(src);
      continue;
    }
    TF_RETURN_IF_ERROR(inspection_required_checker_.IsPlacerInspectionRequired(
        *dst, &needs_inspection));
    if (needs_inspection) {
      inspection_required->insert(dst);
      continue;
    }

    DataType input_type = dst->input_type(edge->dst_input());

    // Colocate two DatasetOp nodes connected by edge of dtype=DT_VARIANT.
    // This is needed to get around the issue in b/135705778.
    if (input_type == DT_VARIANT &&
        data::DatasetOpKernel::IsDatasetOp(&src->op_def()) &&
        data::DatasetOpKernel::IsDatasetOp(&dst->op_def())) {
      TF_RETURN_IF_ERROR(ColocateResourceOrRefEdge(src, dst));
      continue;
    }

    // Even though we can look inside function calling ops, we make an exception
    // here mostly for performance reasons. Looking inside function calling ops
    // is extra overhead. It is only necessary when they return resources. When
    // they don't, we don't look inside them and make this exception here.
    // Looking inside, could potentially enable us to make better placement
    // decisions. It might be worth doing at some point.
    if ((input_type == DT_RESOURCE || IsRefType(input_type)) &&
        !IsExemptFromResourceInputColocation(dst)) {
      TF_RETURN_IF_ERROR(ColocateResourceOrRefEdge(src, dst));
    }
  }

  return Status::OK();
}

namespace {
// Returns tensor list element data type, if the node is one of the ops that
// operate with TensorLists. Otherwise returns DT_INVALID.
DataType GetElementDataType(const Node& node) {
  static absl::flat_hash_set<std::string>* tensor_list_ops =
      new absl::flat_hash_set<std::string>(
          {"TensorListReserve", "TensorListFromTensor", "EmptyTensorList",
           "TensorListSplit", "TensorListScatter", "TensorListScatterV2",
           "TensorListScatterIntoExistingList", "TensorListPushBack",
           "TensorListPushBackBatch", "TensorListPopBack", "TensorListStack",
           "TensorListConcat", "TensorListConcatV2", "TensorListGetItem",
           "TensorListSetItem", "TensorListGather", "TensorListConcatLists"});

  if (tensor_list_ops->contains(node.type_string())) {
    DataType element_type;
    if (GetNodeAttr(node.attrs(), "element_dtype", &element_type).ok()) {
      return element_type;
    }
  }

  return DT_INVALID;
}
}  // namespace

Status ColocationGraph::AddHostOnlyDataTypesConstraints() {
  auto is_variant = [](DataType dtype) -> bool { return dtype == DT_VARIANT; };

  auto is_cpu_device = [](const std::pair<DeviceType, int32>& entry) -> bool {
    return entry.first == DEVICE_CPU;
  };

  for (Node* node : graph_.nodes()) {
    // Skip nodes that do not have DT_VARIANT inputs.
    if (absl::c_none_of(node->input_types(), is_variant)) continue;

    // Skip nodes that can't be placed on GPU anyway.
    Member& root = members_[FindAndUpdateRoot(node->id())];
    if (absl::c_all_of(root.supported_device_types(), is_cpu_device)) continue;

    // Stop DFS traversal when found the underlying data type of a variant.
    absl::optional<bool> is_host_data_type;

    auto edge_filter = [&](const Edge& edge) -> bool {
      // We already found the underlying data type.
      if (is_host_data_type.has_value()) return false;

      // Otherwise follow only DT_VARIANT data edges.
      auto edge_dtype = [&]() -> DataType {
        return edge.src()->output_type(edge.src_output());
      };
      return !edge.IsControlEdge() && edge_dtype() == DT_VARIANT;
    };

    auto enter = [&](Node* n) -> void {
      DataType element_type = GetElementDataType(*n);
      // To handle nested lists continue traversal after finding a TensorList
      // operation that uses DT_VARIANT for element type.
      if (element_type == DT_INVALID || element_type == DT_VARIANT) return;
      is_host_data_type = DataTypeAlwaysOnHost(element_type);
    };

    ReverseDFSFrom(graph_, {node}, enter, /*leave=*/nullptr,
                   /*stable_comparator=*/nullptr, edge_filter);

    if (is_host_data_type.has_value() && *is_host_data_type) {
      VLOG(2) << "Limit node possible devices to CPU only, because it has a "
                 "DT_VARIANT input with host-only underlying data type: "
              << "node=" << node->name();

      // Restrict possible device types to CPU only.
      PossibleDevices possible_devices;
      absl::c_copy_if(root.supported_device_types(),
                      std::back_inserter(possible_devices.device_types),
                      is_cpu_device);

      TF_RETURN_IF_ERROR(root.LimitToPossibleDevices(
          possible_devices, /*allow_soft_placement=*/false));
    }
  }

  return Status::OK();
}

Status ColocationGraph::AddInspectionConstraints(
    const std::unordered_set<Node*>& inspection_required) {
  for (Node* node : inspection_required) {
    IOColocationGroups groups;
    TF_RETURN_IF_ERROR(
        inspecting_placer_.ComputeIOColocationGroups(*node, &groups));
    VLOG(2) << "Computed IOColocationGroups for node " << node->name()
            << ":\n\t" << groups.DebugString();
    TF_RETURN_IF_ERROR(ApplyIOColocationGroups(groups, *node));
  }
  return Status::OK();
}

Status ColocationGraph::Initialize() {
  TF_RETURN_IF_ERROR(InitializeMembers());

  std::unordered_set<Node*> inspection_required;
  TF_RETURN_IF_ERROR(ColocateResourceAndRefEdges(&inspection_required));
  TF_RETURN_IF_ERROR(AddHostOnlyDataTypesConstraints());
  TF_RETURN_IF_ERROR(AddInspectionConstraints(inspection_required));
  TF_RETURN_IF_ERROR(ColocateAllNodes());

  for (Node* node : graph_.op_nodes()) {
    int root_id = FindAndUpdateRoot(node->id());
    members_[root_id].MaybeExcludeXlaDevices();
  }

  return Status::OK();
}

// pair containing a node and whether this node has a resource input
// from the node requiring placer inspection.
using NodeAndBool = std::pair<const Node*, bool>;

namespace {

// Returns a vector of node names from `nodes`.
std::vector<string> NodeAndBoolToString(const std::vector<NodeAndBool>& nodes) {
  std::vector<string> v;
  v.reserve(nodes.size());
  for (const NodeAndBool& node_and_bool : nodes) {
    v.push_back(node_and_bool.first->name());
  }
  return v;
}

// Given a node requiring placer inspection and its IOColocationGroups,
// computes `group_nodes`.
// group_nodes[i] contains the nodes that are members of colocation
// group i. These nodes are inputs or outputs of `node`.
// group_nodes[i][j] is a pair containing a node and whether this node
// has a resource input from `node`.
// Note:
// The same node can be added multiple times to the same group.
// The same node can be added to multiple groups.
Status GetGroupNodes(const IOColocationGroups& groups, const Node& node,
                     std::vector<std::vector<NodeAndBool>>* group_nodes) {
  group_nodes->reserve(groups.group_devices.size());
  for (int arg_idx = 0; arg_idx < groups.input_groups.size(); ++arg_idx) {
    const Node* src;
    TF_RETURN_IF_ERROR(node.input_node(arg_idx, &src));
    int group_id = groups.input_groups[arg_idx];
    (*group_nodes)[group_id].emplace_back(src, false);
  }

  for (const Edge* edge : node.out_edges()) {
    if (edge->IsControlEdge()) {
      continue;
    }

    int group_id = groups.output_groups[edge->src_output()];
    (*group_nodes)[group_id].emplace_back(
        edge->dst(), edge->dst()->input_type(edge->dst_input()) == DT_RESOURCE);
  }

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Colocated inputs/outputs of node: " << node.DebugString();
    for (const std::vector<NodeAndBool>& nodes : *group_nodes) {
      VLOG(2) << "\t[" << absl::StrJoin(NodeAndBoolToString(nodes), "\t\n")
              << "]";
    }
  }
  return Status::OK();
}

// Returns whether the device_type in `device_attributes` is supported.
bool IsSupportedDeviceType(const DeviceAttributes& device_attributes,
                           const DeviceType& supported_type) {
  if (DeviceType(device_attributes.device_type()) == supported_type) {
    return true;
  }
  return IsCompositeDevice(device_attributes.device_type());
}

}  // namespace

Status ColocationGraph::ApplyIOColocationGroups(
    const IOColocationGroups& groups, const Node& node) {
  if (groups.input_groups.size() != node.num_inputs()) {
    return errors::Internal(
        "Cannot apply input/output device constraints to node ",
        node.DebugString(), " because input_groups.size() (",
        groups.input_groups.size(),
        ") is different from number of inputs into the op node (",
        node.num_inputs(), ")");
  }
  if (groups.output_groups.size() != node.num_outputs()) {
    return errors::Internal(
        "Cannot apply input/output device constraints to node ",
        node.DebugString(), " because output_groups.size() (",
        groups.output_groups.size(),
        ") is different from number of outputs into the op node (",
        node.num_outputs(), ")");
  }

  // group_nodes[i] contains the nodes that are members of colocation
  // group i. These nodes are inputs or outputs of `node`.
  // group_nodes[i][j] is a pair containing the node and whether this node
  // has a resource input from `node`.
  // The same node can be added multiple times to the same group.
  // The same node can be added to multiple groups.
  // NOTE: group ids are guarantees to be [0, 1, ..., num_groups].
  std::vector<std::vector<NodeAndBool>> group_nodes(
      groups.group_devices.size());
  TF_RETURN_IF_ERROR(GetGroupNodes(groups, node, &group_nodes));

  // Colocate nodes in each group
  for (const std::vector<NodeAndBool>& nodes : group_nodes) {
    for (int i = 1; i < nodes.size(); ++i) {
      VLOG(2) << "Colocating \"" << nodes[0].first->name() << "\" and \""
              << nodes[i].first->name() << "\"";
      if (nodes[i].second) {
        TF_RETURN_IF_ERROR(
            ColocateResourceOrRefEdge(nodes[0].first, nodes[i].first));
      } else {
        TF_RETURN_IF_ERROR(ColocateNodes(*nodes[0].first, *nodes[i].first));
      }
    }
  }

  // Limit devices in each group
  for (int group_id = 0; group_id < groups.group_devices.size(); ++group_id) {
    // Nothing to do for empty groups. Groups can be empty if some output
    // of an op is not used.
    if (group_nodes[group_id].empty()) {
      continue;
    }
    const Node* group_node = group_nodes[group_id][0].first;
    const PossibleDevices& possible_devices = groups.group_devices[group_id];
    TF_RETURN_IF_ERROR(LimitToPossibleDevices(*group_node, possible_devices));
  }

  return Status::OK();
}

Status ColocationGraph::ColocateNodeToGroup(
    std::unordered_map<StringPiece, const Node*, StringPieceHasher>*
        colocation_group_root,
    const Node* node, StringPiece colocation_group) {
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
      if (!allow_soft_placement_) {
        return AttachDef(s, *node);
      }
      if (log_device_placement_) {
        LOG(INFO) << "Ignoring request to colocate node '" << node->name()
                  << "' with nodes in colocation group '" << colocation_group
                  << "' because soft placement is on and an attempt at doing "
                     "so resulted in the following error: "
                  << AttachDef(s, *node).ToString();
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
Status ColocationGraph::ColocateNodes(const Node& x, const Node& y) {
  int x_root = FindAndUpdateRoot(x.id());
  int y_root = FindAndUpdateRoot(y.id());
  return ColocateNodes(x, x_root, y, y_root);
}

// This overload of ColocateNodes() allows a caller to provide the root node
// ids for the two nodes. For large graphs, this noticeably reduces the
// graph load time.
Status ColocationGraph::ColocateNodes(const Node& x, int x_root, const Node& y,
                                      int y_root) {
  if (x_root == y_root) {
    return Status::OK();
  }

  Member* new_root_member;
  Member* old_root_member;
  Member::Merge(&members_, x_root, y_root, &new_root_member, &old_root_member,
                /*dry_run=*/true);

  // Merge the partial device specifications, and ensure that they are
  // compatible. NULL options_ is treated as allowing soft placement.
  // If there is an error, nothing is modified.
  // TODO(mrry): Consider enriching the error message by pointing
  // out which nodes have the explicit partial device
  // specifications that caused this conflict.
  Status s = new_root_member->MergeDeviceNames(*old_root_member,
                                               allow_soft_placement_);
  if (!s.ok()) {
    return errors::InvalidArgument(
        "Cannot colocate nodes ",
        errors::FormatColocationNodeForError(x.name()), " and ",
        errors::FormatColocationNodeForError(y.name()), ": ",
        s.error_message());
  }

  // Ensure that the common root has at least one supported device
  // type, by computing the intersection of
  // new_root_member.supported_device_types and
  // old_root_member.supported_device_types.
  if (!new_root_member->MergeSupportedDevices(*old_root_member)) {
    return errors::InvalidArgument(
        "Cannot colocate nodes ",
        errors::FormatColocationNodeForError(x.name()), " and ",
        errors::FormatColocationNodeForError(y.name()),
        " because no device type supports both of those nodes and the "
        "other nodes colocated with them.",
        DebugInfo(x_root), DebugInfo(y_root));
  }

  // All error checks are done, merge the colocation graphs.
  Member::Merge(&members_, x_root, y_root, &new_root_member, &old_root_member,
                /*dry_run=*/false);
  return Status::OK();
}

Status ColocationGraph::LimitToAssignedDevice(const Node& node) {
  if (node.assigned_device_name_index() < 0) {
    return errors::Internal(
        "Expected an assigned node as argument to LimitToAssignedDevice but "
        "got: ",
        node.DebugString());
  }
  int root = FindAndUpdateRoot(node.id());
  Member& root_member = members_[root];
  return root_member.AssignDevice(node);
}

void ColocationGraph::GetSoftDeviceCandidates(
    const Node& node, const Member& root_member, int root_id,
    std::vector<Device*>* possible_devices) {
  // Try to find supported devices that don't violate resource devices.
  // The soft_device_name is the same as the requested device name
  // without specifying the device type or ID (if assigned and requested
  // devices does not specify them).
  DeviceNameUtils::ParsedName soft_device_name =
      root_member.GetPreferredSoftDeviceName();
  device_set_.FindMatchingDevices(soft_device_name, possible_devices);
  if (!possible_devices->empty()) {
    *possible_devices = FilterSupportedDevices(
        *possible_devices, root_member.supported_device_types(),
        default_local_device_);
  }

  if (!possible_devices->empty()) {
    return;
  }

  // TODO(iga): Disallow changing resource devices when this ColocationGraph
  // is for :
  // - a function called by an op requiring deep inspection, or
  // - a graph containing ops requiring inspection.
  // It is fairly tricky to make changing resource devices in presence of
  // ops requiring inspection work correctly. One thing it would require is to
  // communicate these "resource movement" decisions across Placer instances.

  // Failed to find supported devices that don't violate resource devices.
  // Try finding some devices that violated resource devices.
  // If we succeed, we will log a warning below.
  soft_device_name = root_member.GetSoftDeviceName();
  device_set_.FindMatchingDevices(soft_device_name, possible_devices);
  if (!possible_devices->empty()) {
    *possible_devices = FilterSupportedDevices(
        *possible_devices, root_member.supported_device_types(),
        default_local_device_);
  }

  if (!possible_devices->empty()) {
    LOG(WARNING)
        << "Failed to place the graph without changing the devices of some "
           "resources. Some of the operations (that had to be colocated with "
           "resource generating operations) are not supported on the "
           "resources' devices. Current candidate devices are [\n  "
        << absl::StrJoin(DevicesToString(*possible_devices), "\n  ")
        << "].\nSee below for details of this colocation group:"
        << DebugInfo(root_id);
  }
}

Status ColocationGraph::LimitToPossibleDevices(const Node& node,
                                               const PossibleDevices& devices) {
  int root = FindAndUpdateRoot(node.id());
  Member& root_member = members_[root];
  return root_member.LimitToPossibleDevices(devices, allow_soft_placement_);
}

Status ColocationGraph::GetDevicesForNode(
    Node* node, const std::vector<Device*>** possible_devices) {
  *possible_devices = nullptr;
  const int node_root = FindAndUpdateRoot(node->id());
  if (!members_[node_root].possible_devices().empty()) {
    *possible_devices = &members_[node_root].possible_devices();
    return Status::OK();
  }

  Member& root_member = members_[node_root];

  // We have not yet computed the possible devices for the
  // colocated node set containing 'node', so we do so now using the
  // constraints on the root node.

  // "devices" will contain the set of feasible placements for the
  // colocated node set containing 'node'.
  // NOTE: Basing possible device computation on requested device name
  // is guaranteed to respect the assigned and resource device names because
  // requested device is always a specialization of both.
  std::vector<Device*> devices;
  if (DeviceNameUtils::HasSomeDetails(root_member.requested_device_name())) {
    // The root node has a (possibly partial) device
    // specification, so enumerate the physical devices that
    // conform to it.
    device_set_.FindMatchingDevices(root_member.requested_device_name(),
                                    &devices);

    if (!devices.empty()) {
      // Filter devices into those that are compatible with the root
      // node (and its children).
      devices = FilterSupportedDevices(
          devices, root_member.supported_device_types(), default_local_device_);
    }

    // Perform soft placement if allow_soft_placement_ is set.
    if (devices.empty() && allow_soft_placement_) {
      GetSoftDeviceCandidates(*node, root_member, node_root, &devices);
    }

    if (devices.empty()) {
      // Return an error when a physical device that matches an explicit
      // device specification is not found. This ensures that we don't
      // assign a node to GPU when the user wanted to force it on CPU.
      string debug_info = DebugInfo(node_root);

      DeviceNameUtils::ParsedName specified_device_name;
      if (DeviceNameUtils::ParseFullName(node->requested_device(),
                                         &specified_device_name) &&
          specified_device_name == root_member.requested_device_name()) {
        // The specified device and merged set device match, and
        // will appear in the GraphDef (for debugging), so just
        // print the specified device.
        std::vector<Device*> devices_matching_nodedef;
        device_set_.FindMatchingDevices(specified_device_name,
                                        &devices_matching_nodedef);
        if (devices_matching_nodedef.empty()) {
          // Sometimes it is almost impossible to understand the problem
          // without a list of available devices.
          std::vector<string> device_names;
          for (const Device* device : device_set_.devices()) {
            device_names.push_back(device->name());
          }
          std::sort(device_names.begin(), device_names.end());

          string gpu_msg = "";
          if (!IsGoogleCudaEnabled() &&
              absl::AsciiStrToLower(specified_device_name.type) == "gpu") {
            gpu_msg =
                " The requested device appears to be a GPU, but CUDA is not "
                "enabled.";
          }

          return errors::InvalidArgument(
              errors::FormatNodeNameForError(node->name()),
              " was explicitly assigned to ", node->requested_device(),
              " but available devices are [ ",
              absl::StrJoin(device_names, ", "), " ]. Make sure ",
              "the device specification refers to a valid device.", gpu_msg);
        } else if (specified_device_name.has_type) {
          return errors::InvalidArgument(
              "Could not satisfy explicit device specification '",
              node->requested_device(), "' because no supported kernel for ",
              specified_device_name.type, " devices is available.", debug_info,
              "\nOp: ", node->type_string(),
              "\nNode attrs: ", node->attrs().DebugString(),
              "\nRegistered kernels:\n",
              KernelsRegisteredForOp(node->type_string()));
        } else {
          return errors::InvalidArgument(
              "Could not satisfy explicit device specification '",
              node->requested_device(), debug_info);
        }
      } else {
        // The specified device may be a valid device but the
        // merged set device is different, so print both.
        // TODO(b/129057603): There are many possibilities at this point.
        // Provide good error messages.
        return errors::InvalidArgument(
            "Could not satisfy explicit device specification '",
            node->requested_device(), "' because the node ",
            errors::FormatColocationNodeForError(node->name()),
            " was colocated with a group of nodes that ",
            "required incompatible device '",
            DeviceNameUtils::ParsedNameToString(
                root_member.requested_device_name()),
            "'. All available devices [",
            absl::StrJoin(DevicesToString(device_set_.devices()), ", "), "]. ",
            debug_info);
      }
    }
  } else {
    // The device is completely unspecified, so enumerate the devices that
    // support all of the nodes in the set.
    if (device_set_.devices().empty()) {
      return errors::Internal("No devices are registered");
    }
    devices = FilterSupportedDevices(device_set_.devices(),
                                     root_member.supported_device_types(),
                                     default_local_device_);

    if (devices.empty()) {
      return errors::InvalidArgument(
          "Node had no OpKernel registered to support this operation: ",
          "Operation was ", node->type_string(), " and inputs were [",
          DataTypeVectorString(node->input_types()), "].\n",
          DebugInfo(node_root));
    }
  }

  // Cache the result of the possible devices for this node group.
  root_member.set_possible_devices(std::move(devices));
  *possible_devices = &root_member.possible_devices();
  return Status::OK();
}

Status ColocationGraph::InitializeMembers() {
  for (Node* node : graph_.op_nodes()) {
    Status status = InitializeMember(*node, &members_[node->id()]);
    if (!status.ok()) {
      return AttachDef(status, *node);
    }
  }
  return Status::OK();
}

string ColocationGraph::DebugString() const {
  std::unordered_set<int> roots;
  std::vector<string> root_strings;
  for (const Node* node : graph_.nodes()) {
    if (!node->IsOp()) {
      continue;
    }
    int node_root = FindRoot(node->id());
    if (roots.count(node_root) == 0) {
      root_strings.push_back(DebugInfo(node_root));
      roots.insert(node_root);
    }
  }
  return absl::StrJoin(root_strings, "\n");
}

// Returns debugging info for the node referred to by 'node_root'.
string ColocationGraph::DebugInfo(const int node_root) const {
  string text(
      "\nColocation Debug Info:\n"
      "Colocation group had the following types and supported devices: ");

  // If this node is part of a colocation group, then we want to
  // collect the mapping of ops to supported devices, so that
  // the user can see why an unsatisfiable placement occurred.

  std::unordered_map<string, string> type_to_devices;
  std::vector<const Node*> colocation_nodes;
  int num_nodes_found = 0;

  for (const Node* node : graph_.nodes()) {
    if (!node->IsOp()) {
      continue;
    }
    int id = node->id();
    if (FindRoot(id) != node_root) {
      continue;
    }
    ++num_nodes_found;
    colocation_nodes.push_back(node);

    PrioritizedDeviceTypeVector supported_types;
    SupportedDeviceTypesForNode(device_types_, node->def(), &supported_types,
                                &local_address_spec_)
        .IgnoreError();
    string devices_registered;
    for (const auto& device_type : supported_types) {
      strings::StrAppend(&devices_registered,
                         DeviceTypeString(device_type.first), " ");
    }

    const string& op_type = node->type_string();
    type_to_devices[op_type] = std::move(devices_registered);
  }
  strings::StrAppend(&text, "\nRoot ", members_[node_root].DebugString());

  for (const auto& td : type_to_devices) {
    strings::StrAppend(&text, "\n", td.first, ": ", td.second);
  }
  strings::StrAppend(&text,
                     "\n\nColocation members, user-requested devices, and "
                     "framework assigned devices, if any:");
  for (const Node* node : colocation_nodes) {
    strings::StrAppend(&text, "\n  ", node->name(), " (", node->type_string(),
                       ") ", node->requested_device());
    if (node->has_assigned_device_name()) {
      strings::StrAppend(
          &text, " framework assigned device=", node->assigned_device_name());
    }
  }
  strings::StrAppend(&text, "\n");

  if (num_nodes_found <= 0) {
    text.clear();
  }
  return text;
}

Status ColocationGraph::InitializeMemberWithAssignedDevice(
    const string& assigned_device_name, const string& node_type,
    Member* member) {
  // This node has already been assigned to a device, so we
  // respect this placement, after sanity-checking it.
  // NOTE: Since any assignment must have been performed by
  // the TensorFlow runtime, we consider errors in this branch to
  // be INTERNAL.
  TF_RETURN_IF_ERROR(member->SetAssignedDeviceName(assigned_device_name));

  // Since assigned device must be a full specification, do extra checks.
  const Device* assigned_device =
      device_set_.FindDeviceByName(assigned_device_name);
  if (assigned_device == nullptr) {
    // TODO(b/129295848, b/122851476): Remove the bit about cross-host function
    // calls when they are supported.
    return errors::Internal(
        "Assigned device '", assigned_device_name,
        "' does not match any device. This error can happen when one attempts "
        "to run a tf.function with resource inputs residing on remote devices. "
        "This use case is currently not supported. Here are the devices "
        "available on this machine: [",
        absl::StrJoin(DevicesToString(device_set_.devices()), ", "), "].",
        "If you are seeing this error when running using a tf.Session, set "
        "share_cluster_devices_in_session to true in the tf.ConfigProto.");
  }

  for (const auto& d : member->supported_device_types()) {
    if (IsSupportedDeviceType(assigned_device->attributes(), d.first)) {
      return Status::OK();
    }
  }

  return errors::Internal("Assigned device '", assigned_device_name,
                          "' does not have registered OpKernel support "
                          "for ",
                          node_type);
}

Status ColocationGraph::InitializeMember(const Node& node, Member* member) {
  TF_RETURN_IF_ERROR(member->SetParentAndSupportedDevices(
      node, device_types_, &local_address_spec_));

  if (node.has_assigned_device_name()) {
    TF_RETURN_IF_ERROR(InitializeMemberWithAssignedDevice(
        node.assigned_device_name(), node.type_string(), member));
  } else {
    // This node has not yet been assigned to a device, so we
    // calculate any constraints due to the set of registered
    // kernels and any (partial) user-provided device specification
    // in the NodeDef.

    // If no kernels are registered for this op type, fail with an error.
    if (member->supported_device_types().empty()) {
      std::set<string> registered_device_types;
      for (Device* d : device_set_.devices()) {
        registered_device_types.insert(d->device_type());
      }
      return errors::InvalidArgument(
          "No OpKernel was registered to support Op '", node.type_string(),
          "' used by ", errors::FormatNodeNameForError(node.name()),
          " with these attrs: [", node.attrs().DebugString(),
          "]\n"
          "Registered devices: [",
          absl::StrJoin(registered_device_types, ", "), "]\n",
          "Registered kernels:\n", KernelsRegisteredForOp(node.type_string()));
    }

    // If the NodeDef contains a device, then we interpret it as a
    // (partial) device specification.
    if (!node.requested_device().empty()) {
      if (IsRefOrResourceGeneratorNode(node)) {
        // Treat requested device on resource generating nodes as assigned
        // device so that we don't override it.
        TF_RETURN_IF_ERROR(member->SetResourceDeviceName(node));
      } else {
        // The user has specified a device in the NodeDef, try to find a
        // valid device matching their specification in the set of
        // devices.
        // NOTE: The full name may specify a device that is not in
        // n.supported_device_types(), but we check that in AssignDevice().
        TF_RETURN_IF_ERROR(member->SetRequestedDeviceName(node));
      }
    }
  }
  return Status::OK();
}

// Returns a list of devices having type in supported_device_types.  The
// returned list is sorted by preferred type (higher numeric type is preferred).
/*static*/ std::vector<Device*> ColocationGraph::FilterSupportedDevices(
    const std::vector<Device*>& devices,
    const PrioritizedDeviceTypeVector& supported_device_types,
    const Device* default_local_device) {
  Device* filtered_default_device = nullptr;
  PrioritizedDeviceVector prioritized_filtered_devices;
  for (const auto& supported_device_type : supported_device_types) {
    for (Device* device : devices) {
      if (IsSupportedDeviceType(device->attributes(),
                                supported_device_type.first)) {
        if (default_local_device &&
            (device == default_local_device ||
             // TODO(nareshmodi, fishx): At times the device pointer in the
             // device set is different to the one passed in as the default
             // device. Figure out why this might be.
             device->name() == default_local_device->name())) {
          filtered_default_device = device;
        } else {
          prioritized_filtered_devices.emplace_back(
              device, supported_device_type.second);
        }
      }
    }
  }
  DeviceSet::SortPrioritizedDeviceVector(&prioritized_filtered_devices);

  std::vector<Device*> filtered_devices;
  if (filtered_default_device != nullptr) {
    filtered_devices.emplace_back(filtered_default_device);
  }
  for (const auto& prioritized_filtered_device : prioritized_filtered_devices) {
    filtered_devices.push_back(prioritized_filtered_device.first);
  }
  return filtered_devices;
}

}  // namespace tensorflow
