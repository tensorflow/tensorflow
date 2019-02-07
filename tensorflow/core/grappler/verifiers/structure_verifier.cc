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

#include "tensorflow/core/grappler/verifiers/structure_verifier.h"

#include <string>
#include <vector>

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/validate.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/grappler/verifiers/graph_verifier.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

// TODO(ashwinm): Expand this to add more structural checks.
Status StructureVerifier::Verify(const GraphDef& graph) {
  StatusGroup status_group;

  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             graph.library());
  status_group.Update(tensorflow::graph::ValidateGraphDefAgainstOpRegistry(
      graph, function_library));
  status_group.Update(tensorflow::graph::VerifyNoDuplicateNodeNames(graph));

  std::vector<const NodeDef*> topo_order;
  status_group.Update(ComputeTopologicalOrder(graph, &topo_order));
  return status_group.as_status();
}

}  // end namespace grappler
}  // end namespace tensorflow
