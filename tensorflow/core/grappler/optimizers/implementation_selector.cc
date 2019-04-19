/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/implementation_selector.h"

#include <string>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/function_api_info.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {

Status UpdateNodeDef(NodeDef* node_def, const string& funcName,
                     const FunctionApiInfo& apiInfo) {
  VLOG(3) << "Node def before swap is: " << node_def->DebugString();
  auto tin = node_def->mutable_attr()->find("Tin");
  tin->second.mutable_list()->clear_type();
  for (const auto& tin_dtype : apiInfo.input_arg_dtypes()) {
    tin->second.mutable_list()->add_type(tin_dtype);
  }

  auto tout = node_def->mutable_attr()->find("Tout");
  tout->second.mutable_list()->clear_type();
  for (const auto& tout_dtype : apiInfo.output_arg_dtypes()) {
    tout->second.mutable_list()->add_type(tout_dtype);
  }

  if (apiInfo.function_type() == FunctionApiInfo::BACKWARD) {
    // Update the inputs since for backward function, it might have different
    // number of inputs due the different number output from forward function.
    // The output of forward function are composed by two parts:
    //   1. Real output tensors from defun.
    //   2. Internal states that will be used for gradient calculation.
    // Part 1 will be static, and part 2 could be different based on the
    // different implementation.

    const int prev_input_size = node_def->input_size();
    const int diff = prev_input_size - apiInfo.input_arg_dtypes().size();
    if (diff >= 0) {
      for (int i = 0; i < diff; ++i) node_def->mutable_input()->RemoveLast();
    } else {
      // Adding new inputs for internal states, the name of the internal states
      // should be in format "{forward_node_name}:{index}", where the newly
      // added index should start from last index of the state.
      // Eg:
      // {
      //   input: "gradients/unified_lstm/strided_slice_1_grad/StridedSliceGrad"
      //   input: "gradients/zeros_like_1"
      //   input: "gradients/zeros_like_2"
      //   input: "unified_lstm/StatefulPartitionedCall:3"
      //   input: "unified_lstm/StatefulPartitionedCall:4"
      //   # New input should be "unified_lstm/StatefulPartitionedCall:5"
      // }
      const string last_input = node_def->input(prev_input_size - 1);
      const std::vector<string> name_index = ::absl::StrSplit(last_input, ':');
      if (name_index.size() != 2) {
        return errors::InvalidArgument(
            "Invalid format of input node name: ", last_input,
            " Expected: {forward_node_name}:{index}");
      }
      const absl::string_view node_name = name_index[0];
      int last_index;
      if (!::absl::SimpleAtoi(name_index[1], &last_index)) {
        return errors::InvalidArgument(
            "The index of input node is expected to be number, got: ",
            name_index[1]);
      }
      for (int i = 1; i <= -diff; ++i)
        node_def->add_input(strings::StrCat(node_name, ":", i + last_index));
    }
  }

  node_def->mutable_attr()->find("f")->second.mutable_func()->set_name(
      funcName);

  VLOG(3) << "Node def after swap is: " << node_def->DebugString();
  return Status::OK();
}

Status ImplementationSelector::LoadFunctions(const GraphDef& graph) {
  lib_info_.reset(new FunctionLibraryApiInfo);
  TF_RETURN_IF_ERROR(lib_info_->Init(graph.library()));
  return Status::OK();
}

Status ImplementationSelector::MaybeOptimizeFunctionCall(
    NodeDef* node_def) const {
  // There are two ways of calling functions:
  //  1. By specifying an op name as a function name, or
  //  2. Via the @defun functional interface, where the real function call
  //     happens with partitionedcall op, and the function name appear as the
  //     attribute with name "f" and type func. In this use case, there are more
  //     attributes need to be taken care, like Tin and Tout which take care of
  //     the DTYPE of input/output.
  std::vector<string> function_attribute_names;
  for (const auto& attr : node_def->attr()) {
    if (attr.second.has_func() &&
        lib_info_->GetApiInfo(attr.second.func().name()) != nullptr) {
      function_attribute_names.emplace_back(attr.first);
    }
  }

  if (function_attribute_names.empty() &&
      lib_info_->GetApiInfo(node_def->op()) == nullptr) {
    // A regular op, or a function which has no interface.
    return Status::OK();
  }

  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(node_def->device(), &parsed_name) ||
      !parsed_name.has_type) {
    return errors::Internal("Could not parse device name:", node_def->device());
  }
  VLOG(2) << "Op " << node_def->name() << " runs on " << node_def->device()
          << " = (" << parsed_name.type << ")";

  for (const auto& attr_name : function_attribute_names) {
    string function_name = node_def->attr().at(attr_name).func().name();
    std::vector<string> equiv_func_names;
    TF_RETURN_IF_ERROR(lib_info_->GetEquivalentImplementations(
        function_name, &equiv_func_names));
    for (const auto& func_name : equiv_func_names) {
      const auto& func_api_info = lib_info_->GetApiInfo(func_name);
      if (func_api_info->preferred_device() == parsed_name.type) {
        VLOG(2) << "Swapping: " << function_name << " TO: " << func_name;
        TF_RETURN_IF_ERROR(UpdateNodeDef(node_def, func_name, *func_api_info));
        break;
      }
    }
  }

  if (lib_info_->GetApiInfo(node_def->op()) != nullptr) {
    std::vector<string> equiv_func_names;
    TF_RETURN_IF_ERROR(lib_info_->GetEquivalentImplementations(
        node_def->op(), &equiv_func_names));
    for (const string& func_name : equiv_func_names) {
      const auto func_api_info = lib_info_->GetApiInfo(func_name);
      if (func_api_info->preferred_device() == parsed_name.type) {
        node_def->set_op(func_name);
        break;
      }
    }
  }
  return Status::OK();
}

Status ImplementationSelector::SelectImplementation(GraphDef* graph) const {
  if (!graph->has_library()) {
    VLOG(2) << "Skipping graph since it does not have function def";
    return Status::OK();
  }
  if (lib_info_->empty()) {
    VLOG(2) << "Skipping optimization since lib_info is empty";
    return Status::OK();
  }

  for (int k = 0; k < graph->node_size(); ++k)
    TF_RETURN_IF_ERROR(MaybeOptimizeFunctionCall(graph->mutable_node(k)));

  return Status::OK();
}

Status ImplementationSelector::Optimize(Cluster* cluster,
                                        const GrapplerItem& item,
                                        GraphDef* optimized_graph) {
  *optimized_graph = item.graph;
  TF_RETURN_IF_ERROR(LoadFunctions(*optimized_graph));
  return SelectImplementation(optimized_graph);
}

}  // end namespace grappler
}  // end namespace tensorflow
