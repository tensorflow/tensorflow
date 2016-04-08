/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/graph/validate.h"

#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace graph {

Status ValidateGraphDef(const GraphDef& graph_def,
                        const OpRegistryInterface& op_registry) {
  Status s;
  for (const NodeDef& node_def : graph_def.node()) {
    // Look up the OpDef for the node_def's op name.
    const OpDef* op_def = op_registry.LookUp(node_def.op(), &s);
    TF_RETURN_IF_ERROR(s);
    TF_RETURN_IF_ERROR(ValidateNodeDef(node_def, *op_def));
  }

  return s;
}

Status ValidateGraphDefAgainstOpRegistry(
    const GraphDef& graph_def, const OpRegistryInterface& op_registry) {
  GraphDef copy(graph_def);
  TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(&copy, op_registry, 0));
  return ValidateGraphDef(copy, op_registry);
}

Status ValidateGraphDefAgainstOpList(const GraphDef& graph_def,
                                     const OpList& op_list) {
  OpListOpRegistry registry(&op_list);
  return ValidateGraphDefAgainstOpRegistry(graph_def, registry);
}

void GetOpListForValidation(OpList* op_list, const OpRegistry& op_registry) {
  op_registry.Export(false, op_list);
  RemoveDescriptionsFromOpList(op_list);
}

}  // namespace graph
}  // namespace tensorflow
