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
#include "tensorflow/core/common_runtime/inspecting_placer.h"

#include <memory>
#include <unordered_map>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/colocation_graph.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/common_runtime/placer_inspection_required_ops_utils.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

string IOColocationGroups::DebugString() const {
  std::unordered_map<int, std::vector<string>> group_members;
  for (int arg_index = 0; arg_index < input_groups.size(); ++arg_index) {
    int group_id = input_groups[arg_index];
    group_members[group_id].push_back(strings::StrCat("i:", arg_index));
  }
  for (int ret_index = 0; ret_index < output_groups.size(); ++ret_index) {
    int group_id = output_groups[ret_index];
    group_members[group_id].push_back(strings::StrCat("o:", ret_index));
  }

  std::vector<string> group_strings;
  for (const auto& it : group_members) {
    int group_id = it.first;
    const std::vector<string>& members = it.second;
    const PossibleDevices& devices = group_devices[group_id];
    group_strings.push_back(strings::StrCat(
        "Group(", group_id, " members = [", absl::StrJoin(members, ", "),
        "] requested_device_name = \"",
        DeviceNameUtils::ParsedNameToString(devices.requested_device_name),
        "\" resource_device_name = \"",
        DeviceNameUtils::ParsedNameToString(devices.resource_device_name),
        "\" device_types = [",
        absl::StrJoin(
            devices.device_types, ", ",
            [](string* out, const std::pair<DeviceType, int32>& type_and_pref) {
              out->append(DeviceTypeString(type_and_pref.first));
            }),
        "])"));
  }

  return absl::StrJoin(group_strings, "\n\t");
}

// Utility class for constructing IOColocationGroups from a ColocationGraph.
class ColocationGraphToIOColocationGroups {
 public:
  // colocation_graph is mutable because finding root nodes can update
  // parent pointers. It is not modified otherwise.
  explicit ColocationGraphToIOColocationGroups(
      ColocationGraph* colocation_graph)
      : colocation_graph_(colocation_graph), next_group_id_(0) {}

  void AssignGroups(const gtl::InlinedVector<Node*, 4>& nodes,
                    std::vector<int>* groups) {
    for (int i = 0; i < nodes.size(); ++i) {
      int root_id = colocation_graph_->FindAndUpdateRoot(nodes[i]->id());
      const auto& it = group_ids_.find(root_id);
      int assigned_group_id;
      if (it == group_ids_.end()) {
        group_ids_[root_id] = next_group_id_;
        assigned_group_id = next_group_id_;
        ++next_group_id_;
      } else {
        assigned_group_id = it->second;
      }
      groups->push_back(assigned_group_id);
    }
  }

  Status FillGroups(std::vector<PossibleDevices>* group_devices) {
    group_devices->resize(group_ids_.size());
    for (const auto& it : group_ids_) {
      int assigned_group_id = it.second;
      PossibleDevices& possible_devices = (*group_devices)[assigned_group_id];
      const Member& member = colocation_graph_->members()[it.first];
      TF_RETURN_IF_ERROR(member.FillPossibleDevices(&possible_devices));
    }
    return Status::OK();
  }

 private:
  ColocationGraph* colocation_graph_;
  // Allocated group ids: collocation_graph root id -> allocated group id.
  std::unordered_map<int, int> group_ids_;
  int next_group_id_;
};

InspectingPlacer::InspectingPlacer(const FunctionStack& stack,
                                   const FunctionLibraryDefinition* flib_def,
                                   const DeviceSet* device_set,
                                   const Device* default_device,
                                   bool allow_soft_placement,
                                   bool log_device_placement)
    : stack_(stack),
      flib_def_(*flib_def),
      device_set_(*device_set),
      default_device_(default_device),
      allow_soft_placement_(allow_soft_placement),
      log_device_placement_(log_device_placement) {}

Status InspectingPlacer::ComputeIOColocationGroups(const Node& node,
                                                   IOColocationGroups* groups) {
  const FunctionDef* fdef;
  NameAttrList func;
  TF_RETURN_IF_ERROR(GetFunctionDefAndAttrs(flib_def_, node, &fdef, &func));
  std::unique_ptr<FunctionBody> fbody;

  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*fdef, AttrSlice(&func.attr()),
                                             &flib_def_, &fbody));

  TF_RETURN_IF_ERROR(
      IsolatePlacerInspectionRequiredOps(flib_def_, fbody->graph));
  if (stack_.HasFunction(func.name())) {
    return errors::Unimplemented(
        "Recursive function calls are not supported. Node ",
        FormatNodeForError(node), " inside the body of ",
        errors::FormatFunctionForError(stack_.current_function_name()),
        " calls function ", errors::FormatFunctionForError(func.name()),
        " which is already present in the call stack:\n  ",
        stack_.FormatForError());
  }

  ColocationGraph colocation_graph(
      fbody->graph, stack_.Push(&node, func.name()), &flib_def_, &device_set_,
      default_device_, allow_soft_placement_, log_device_placement_);
  TF_RETURN_IF_ERROR(colocation_graph.Initialize());

  ColocationGraphToIOColocationGroups converter(&colocation_graph);
  converter.AssignGroups(fbody->arg_nodes, &groups->input_groups);
  converter.AssignGroups(fbody->ret_nodes, &groups->output_groups);
  TF_RETURN_IF_ERROR(converter.FillGroups(&groups->group_devices));
  return Status::OK();
}

}  // namespace tensorflow
