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

#include "tensorflow/core/kernels/data/serialization_utils.h"

#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/data/captured_function.h"

namespace tensorflow {
namespace data {

namespace {

// FindStatefulOps searches `graph_def` for all of its stateful ops storing
// their names in `stateful_op_names`.
Status FindStatefulOps(const GraphDef& graph_def,
                       std::vector<string>* stateful_op_names) {
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), graph_def.library());

  // Iterate over all nodes in the graph.
  for (const auto& node : graph_def.node()) {
    // Each Dataset graph has a _Retval op in the end which is marked stateful
    if (node.op() == FunctionLibraryDefinition::kRetOp) continue;
    if (!IsNodeStateful(lib_def, node).ok()) {
      stateful_op_names->push_back(node.op());
    }
  }

  // Iterate over all functions.
  for (const auto& fdef : graph_def.library().function()) {
    if (!fdef.signature().is_stateful()) continue;
    for (const auto& node : fdef.node_def()) {
      if (!IsNodeStateful(lib_def, node).ok()) {
        stateful_op_names->push_back(
            absl::StrCat(node.op(), " in function: ", fdef.signature().name()));
      }
    }
  }
  return Status::OK();
}

}  // namespace

Status AsGraphDef(OpKernelContext* ctx, const DatasetBase* dataset,
                  SerializationContext&& serialization_ctx,
                  GraphDef* graph_def) {
  if (serialization_ctx.external_state_policy() ==
      SerializationContext::ExternalStatePolicy::kFail) {
    TF_RETURN_IF_ERROR(dataset->CheckExternalState());
  }
  if (serialization_ctx.external_state_policy() ==
      SerializationContext::ExternalStatePolicy::kWarn) {
    std::vector<string> stateful_op_names;
    TF_RETURN_IF_ERROR(FindStatefulOps(*graph_def, &stateful_op_names));
    if (!stateful_op_names.empty()) {
      LOG(WARNING)
          << "We found the following stateful ops in the dataset "
             "construction graph whose state would not be serialized and might "
             "cause subtle bugs: "
          << absl::StrJoin(stateful_op_names, ", ");
    }
  }
  GraphDefBuilder b;
  DatasetBase::DatasetGraphDefBuilder db(&b);
  Node* output_node = nullptr;
  TF_RETURN_IF_ERROR(
      db.AddInputDataset(&serialization_ctx, dataset, &output_node));
  // Insert a purely symbolic _Retval node to indicate to consumers which node
  // represents `dataset`.
  ops::UnaryOp("_Retval", output_node,
               b.opts()
                   .WithName("dataset")
                   .WithAttr("T", DT_VARIANT)
                   .WithAttr("index", 0));
  TF_RETURN_IF_ERROR(b.ToGraphDef(graph_def));
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
