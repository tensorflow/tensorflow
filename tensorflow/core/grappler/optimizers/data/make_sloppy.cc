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

#include "tensorflow/core/grappler/optimizers/data/make_sloppy.h"

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"

namespace tensorflow {
namespace grappler {

Status MakeSloppy::Optimize(Cluster* cluster, const GrapplerItem& item,
                            GraphDef* output) {
  *output = item.graph;
  MutableGraphView graph(output);

  for (NodeDef& node : *output->mutable_node()) {
    if (node.op() == "ParallelInterleaveDatasetV2" ||
        node.op() == "ParallelMapDataset" ||
        node.op() == "ParseExampleDataset") {
      (*node.mutable_attr())["sloppy"].set_b(true);
    }
  }
  return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(MakeSloppy, "make_sloppy");

}  // namespace grappler
}  // namespace tensorflow
