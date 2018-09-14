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

REGISTER_GRAPH_OPTIMIZER(ExperimentalImplementationSelector);

Status ExperimentalImplementationSelector::LoadFunctions(
    const GraphDef& graph) {
  lib_info_.reset(new FunctionLibraryApiInfo);
  TF_RETURN_IF_ERROR(lib_info_->Init(graph.library()));
  return Status::OK();
}

Status ExperimentalImplementationSelector::MaybeOptimizeFunctionCall(
    NodeDef* node_def) const {
  const FunctionApiInfo* info = lib_info_->GetApiInfo(node_def->op());
  if (info == nullptr) {
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

  string best_function_name;
  lib_info_->GetBestImplementation(node_def->op(), parsed_name.type,
                                   &best_function_name);
  if (node_def->op() != best_function_name) {
    // The current implementation is not the best, swap the op to the best one.
    // There will be duplicates in the graph and they will be pruned by other
    // grappler plugin since no other node is using their output as inputs.
    // TODO(scottzhu): Update the tf.eager.defun to register functions without
    // having to call them with input data. That will reduce the graph size and
    // save the work for prune them.
    node_def->set_op(best_function_name);
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
