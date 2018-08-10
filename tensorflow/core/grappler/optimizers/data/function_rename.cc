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

#include "tensorflow/core/grappler/optimizers/data/function_rename.h"

#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {

Status FunctionRename::Optimize(Cluster* cluster, const GrapplerItem& item,
                                GraphDef* output) {
  *output = item.graph;
  GraphView graph(output);
  int n = output->mutable_library()->function_size();
  for (int i = 0; i < n; ++i) {
    FunctionDef* fn = output->mutable_library()->mutable_function(i);
    fn->mutable_signature()->set_name(fn->signature().name() + "world");
  }

  return Status::OK();
}

void FunctionRename::Feedback(Cluster* cluster, const GrapplerItem& item,
                              const GraphDef& optimize_output, double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(FunctionRename, "_test_only_function_rename");

}  // end namespace grappler
}  // end namespace tensorflow
