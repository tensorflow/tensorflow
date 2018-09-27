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

#include "tensorflow/core/grappler/optimizers/experimental_implementation_selector.h"

#include <string>

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

Status ExperimentalImplementationSelector::LoadFunctions(
    const GraphDef& graph) {
  lib_info_.reset(new FunctionLibraryApiInfo);
  TF_RETURN_IF_ERROR(lib_info_->Init(graph.library()));
  return Status::OK();
}

Status ExperimentalImplementationSelector::MaybeOptimizeFunctionCall(
    NodeDef* node_def) const {
  // There are two ways of calling functions:
  //  1. By specifying an op name as a function name, or
  //  2. Via the @defun functional interface, where the real function name
  //     appear as the attribute with type func.
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

  string task, device;
  if (!DeviceNameUtils::SplitDeviceName(node_def->device(), &task, &device)) {
    return errors::Internal("Could not split device name:", node_def->device());
  }
  VLOG(2) << "Op " << node_def->name() << " runs on " << node_def->device()
          << " = (" << task << ", " << device << ")";
  DeviceNameUtils::ParsedName parsed_name;
  DeviceNameUtils::ParseLocalName(device, &parsed_name);

  for (const auto& attr_name : function_attribute_names) {
    string function_name = node_def->attr().at(attr_name).func().name();
    string best_function_name;
    lib_info_->GetBestImplementation(function_name, parsed_name.type,
                                     &best_function_name);
    if (function_name != best_function_name) {
      node_def->mutable_attr()
          ->find(attr_name)
          ->second.mutable_func()
          ->set_name(best_function_name);
    }
  }
  if (lib_info_->GetApiInfo(node_def->op()) != nullptr) {
    string best_function_name;
    lib_info_->GetBestImplementation(node_def->op(), parsed_name.type,
                                     &best_function_name);
    if (node_def->op() != best_function_name) {
      node_def->set_op(best_function_name);
    }
  }
  return Status::OK();
}

Status ExperimentalImplementationSelector::SelectImplementation(
    GraphDef* graph) const {
  for (int k = 0; k < graph->node_size(); ++k)
    TF_RETURN_IF_ERROR(MaybeOptimizeFunctionCall(graph->mutable_node(k)));

  return Status::OK();
}

Status ExperimentalImplementationSelector::Optimize(Cluster* cluster,
                                                    const GrapplerItem& item,
                                                    GraphDef* optimized_graph) {
  *optimized_graph = item.graph;
  TF_RETURN_IF_ERROR(LoadFunctions(*optimized_graph));
  return SelectImplementation(optimized_graph);
}

}  // end namespace grappler
}  // end namespace tensorflow
